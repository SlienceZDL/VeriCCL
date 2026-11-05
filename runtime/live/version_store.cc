#include "version_store.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vericcl {
WorkSpec WorkSpec::parse(const std::string& text,int rank,int ranks) {
  WorkSpec s;s.manifest=strict_json(text,128*1024*1024);auto& j=s.manifest;
  const std::set<std::string> keys={"schema_version","version_id","xml_sha256","xml_path","contract_digest","collective","datatype","reduction_op","root","inplace","rank_count","input_size_bytes","slice_bytes","runtime_bytes","channels","ranks","trace_prefix","tolerances","light_telemetry"};
  exact_keys(j,keys,keys);
  if(j.at("schema_version")!=1 || j.at("rank_count")!=ranks || ranks<1 || rank<0 || rank>=ranks)
    throw std::invalid_argument("manifest ABI or rank membership mismatch");
  s.ranks=ranks;s.rank=rank;s.version=j.at("version_id").get<std::string>();s.contract=j.at("contract_digest").get<std::string>();
  s.xml=j.at("xml_path").get<std::string>();s.collective=j.at("collective").get<std::string>();s.datatype=j.at("datatype").get<std::string>();
  s.reduction=j.at("reduction_op").is_null()?"sum":j.at("reduction_op").get<std::string>();
  s.root=j.at("root").is_null()?-1:int(integer(j.at("root"),"root",ranks-1));
  if(!j.at("inplace").is_boolean())throw std::invalid_argument("inplace must be boolean");s.inplace=j.at("inplace");
  s.trace_prefix=j.at("trace_prefix").get<std::string>();
  if(!j.at("light_telemetry").is_boolean())throw std::invalid_argument("light_telemetry must be boolean");s.light_telemetry=j.at("light_telemetry");
  if(s.version.size()!=64 || s.version!=j.at("xml_sha256") || s.xml.empty() || s.xml[0]!='/')
    throw std::invalid_argument("manifest XML identity or path invalid");
  if(sha256(read_file(s.xml))!=s.version)throw std::invalid_argument("XML content hash mismatch");
  std::map<std::string,uint64_t> types={{"float16",2},{"float32",4},{"float64",8},{"bfloat16",2},{"int8",1},{"uint8",1},{"int32",4},{"uint32",4},{"int64",8},{"uint64",8}};
  if(!types.count(s.datatype))throw std::invalid_argument("unsupported datatype");s.element_bytes=types.at(s.datatype);
  if(!std::set<std::string>{"broadcast","reduce","allgather","allreduce","alltoall","reduce_scatter"}.count(s.collective))throw std::invalid_argument("unsupported collective");
  bool reducing=s.collective=="reduce" || s.collective=="allreduce" || s.collective=="reduce_scatter";
  bool rooted=s.collective=="broadcast" || s.collective=="reduce";
  if(rooted!=(s.root>=0) || reducing==j.at("reduction_op").is_null())throw std::invalid_argument("root/reduction fields do not match collective");
  if(!std::set<std::string>{"sum","prod","min","max","avg"}.count(s.reduction))throw std::invalid_argument("unsupported reduction");
  if(s.reduction=="avg" && s.datatype!="float16" && s.datatype!="float32" && s.datatype!="float64" && s.datatype!="bfloat16")
    throw std::invalid_argument("this native runtime supports avg only for floating point datatypes");
  s.message_bytes=integer(j.at("input_size_bytes"),"input_size_bytes");s.slice_bytes=integer(j.at("slice_bytes"),"slice_bytes");
  s.runtime_bytes=integer(j.at("runtime_bytes"),"runtime_bytes");s.channels=int(integer(j.at("channels"),"channels",16));
  if(!s.message_bytes || !s.slice_bytes || !s.channels || s.message_bytes%s.slice_bytes || s.slice_bytes%s.element_bytes)
    throw std::invalid_argument("invalid whole-slice geometry");
  if((s.collective=="alltoall" || s.collective=="reduce_scatter") && (s.message_bytes/s.slice_bytes)%ranks)
    throw std::invalid_argument("collective slices must divide rank count");
  if(s.runtime_bytes!=(s.collective=="allgather"?multiply(s.message_bytes,ranks):s.message_bytes))throw std::invalid_argument("runtime count conversion mismatch");
  Json contract;
  for(const char* name:{"collective","datatype","reduction_op","root","inplace","rank_count","input_size_bytes"})contract[name]=j.at(name);
  if(sha256(contract.dump())!=s.contract)throw std::invalid_argument("external contract digest mismatch");
  const auto& members=j.at("ranks");
  if(!members.is_array() || members.size()!=size_t(ranks))throw std::invalid_argument("manifest rank geometry incomplete");
  for(int r=0;r<ranks;r++) {
    const auto& item=members[r];
    const std::set<std::string> fields={"rank","input_bytes","output_bytes","scratch_bytes","input_offset_bytes","output_offset_bytes","input_read_only","threadblocks","expected_steps","trace_records","steps"};
    exact_keys(item,fields,fields);
    if(item.at("rank")!=r || !item.at("input_read_only").is_boolean())throw std::invalid_argument("manifest rank identity mismatch");
    uint64_t in=integer(item.at("input_bytes"),"input_bytes"),out=integer(item.at("output_bytes"),"output_bytes");
    uint64_t expected=s.message_bytes;
    if(s.collective=="allgather")expected=multiply(expected,ranks);
    if(s.collective=="reduce_scatter")expected/=ranks;
    if(s.collective=="reduce" && r!=s.root)expected=0;
    if(in!=s.message_bytes || out!=expected)throw std::invalid_argument("manifest external buffer sizes mismatch");
    uint64_t io=integer(item.at("input_offset_bytes"),"input_offset_bytes"),oo=integer(item.at("output_offset_bytes"),"output_offset_bytes");
    if(io!=(s.inplace && s.collective=="allgather"?multiply(r,s.message_bytes):0) ||
       oo!=(s.inplace && s.collective=="reduce_scatter"?multiply(r,s.message_bytes/ranks):0))throw std::invalid_argument("in-place partition mismatch");
    auto tb=integer(item.at("threadblocks"),"threadblocks",216),steps=integer(item.at("expected_steps"),"expected_steps",216*256);
    auto records=integer(item.at("trace_records"),"trace_records",uint64_t(INT64_MAX)/80);
    if(records && (records<std::max(uint64_t(1),steps) || s.trace_prefix.empty() || s.trace_prefix[0]!='/'))throw std::invalid_argument("invalid trace capacity/prefix");
    if(!item.at("steps").is_array() || item.at("steps").size()!=steps)throw std::invalid_argument("step metadata incomplete");
    std::set<std::pair<int,int>> positions;
    for(const auto& step:item.at("steps")) {
      const std::set<std::string> f={"tb_id","step_index","kind","channel","send_peer","recv_peer","operation_id","payload_bytes"};exact_keys(step,f,f);
      int t=int(integer(step.at("tb_id"),"tb_id",tb?tb-1:0)),i=int(integer(step.at("step_index"),"step_index",255));
      if(!positions.insert({t,i}).second)throw std::invalid_argument("duplicate native step position");
      integer(step.at("channel"),"channel",15);integer(step.at("payload_bytes"),"payload_bytes");
    }
    if(r==rank) {
      s.local=item;s.buffers={in,out,oo,io,s.inplace};s.input_read_only=item.at("input_read_only");
      s.scratch_bytes=integer(item.at("scratch_bytes"),"scratch_bytes");s.threadblocks=tb;s.expected_steps=steps;s.trace_records=records;
    }
  }
  exact_keys(j.at("tolerances"),{"absolute","relative"});
  for(auto it=j.at("tolerances").begin();it!=j.at("tolerances").end();++it)
    if(!it.value().is_number() || !std::isfinite(it.value().get<double>()) || it.value().get<double>()<0)throw std::invalid_argument("invalid numerical tolerance");
  return s;
}
static void write_new(const std::string& path,const std::string& data) {
  int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
  if(fd<0)throw std::runtime_error("cannot exclusively create evidence: "+path);
  size_t offset=0;
  while(offset<data.size()) {ssize_t n=write(fd,data.data()+offset,data.size()-offset);if(n<=0){close(fd);throw std::runtime_error("evidence write failed");}offset+=n;}
  if(fsync(fd)){close(fd);throw std::runtime_error("evidence fsync failed");}close(fd);
}
std::string immutable_xml_copy(const WorkSpec& spec,const std::string& directory) {
  auto data=read_file(spec.xml);if(sha256(data)!=spec.version)throw std::invalid_argument("XML changed before native preparation");
  auto path=directory+"/version-"+spec.version+".xml";
  struct stat st{};
  if(lstat(path.c_str(),&st)==0) {
    if(!S_ISREG(st.st_mode) || st.st_uid!=getuid() || sha256(read_file(path))!=spec.version)throw std::invalid_argument("immutable version path conflict");
  } else {write_new(path,data);if(chmod(path.c_str(),0400))throw std::runtime_error("cannot protect immutable XML copy");}
  return path;
}
void atomic_evidence(const std::string& path,const Json& value) {
  auto partial=path+".partial";write_new(partial,value.dump(2)+"\n");
  if(rename(partial.c_str(),path.c_str()))throw std::runtime_error("evidence publish failed");
  auto slash=path.rfind('/');if(slash!=std::string::npos){int fd=open(path.substr(0,slash).c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);if(fd<0 || fsync(fd)){if(fd>=0)close(fd);throw std::runtime_error("evidence directory fsync failed");}close(fd);}
}
}
