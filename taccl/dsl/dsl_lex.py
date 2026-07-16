import ply.lex as lex

# Primitive
reserved = {
    'link_type': 'LINKTYPE',
    'define': 'DEFINE',
    'intra_node': 'INTRA_NODE',
    'inter_node': 'INTER_NODE',
    'switch' :'SWITCH',
    'direct' :'DIRECT',
    'match' :'MATCH'
}

# Tokens
tokens = [ 'EQUAL', 'NUMBER', 'ID', 'PREFER', 'HATE', 'CONSTRAINT',  'WAYPOINT', 'DECIMAL' ]\
         + list(reserved.values())

# Operator
t_PREFER = r'\>\>'
t_HATE = r'\<\<'
t_CONSTRAINT = r'\=\>'
t_EQUAL = r'\=\='
t_WAYPOINT = r'\-\>'

literals = "(){}[]<>,./=&"

# Ignore space
t_ignore = " \t"


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value in reserved:
        t.type = reserved[t.value]
    return t

def t_NUMBER(t):
    r'\d+(\.\d+)?'
    t.value = int(t.value) if '.' not in t.value else float(t.value)
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

# lexer = lex.lex()

"""
# -------------------- test --------------------

source = '''
link_tpye nvlink = (25, 0.05)
link_type pcie = (16, 0.1)

define ngpu_per_node = 8
define nnic_per_node = 8
intra_node intra_node_bw_delay = {switch => [((0,1,2,3,4,5,6,7)->(nvlink,1))]}

link_type intra_rtsw_single_node = (12.5, 0.2)
link_type intra_rtsw_inter_node = (12.5, 0.5)
link_type inter_rtsw = (12.5, 10)
link_type intra_ctsw = (12.5, 50)
link_type inter_ctsw = (12.5, 250)

inter_node inter_node_bw_delay = {}

define ctsw = 16
define rtsw = 96
define nnode = 192
'''

lexer = lex.lex()

lexer.input(source)
while True:
    tok = lexer.token()
    if not tok: break      # No more input
    print(tok)
"""
