link_type nvlink = (25, 2)
link_type pcie = (16, 2.25)
define ngpu_per_node = 8
define nnic_per_node = 8
define ctsw = 0
define rtsw = 1
define nnode = 2
intra_node intra_node_bw_delay = {switch => [(0,1,2,3,4,5,6,7)->(nvlink,2)]}
link_type intra_rtsw = (12.5, 80)
link_type intra_ctsw = (12.5, 160)
link_type inter_ctsw = (12.5, 320)
inter_node inter_node_bw_delay = [match => (0,1,2,3,4,5,6,7)]