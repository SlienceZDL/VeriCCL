import ply.yacc as yacc
from taccl.dsl.dsl_lex import *
from taccl.dsl.dsl_auto_ast import *

linktype_v = LinkTypeVisitor()
define_v = DefinitionVisitor()
intra_v = IntraNodeVisitor()
inter_v = InterNodeVisitor()

start = 'def'

def p_empty(p):
    """empty :"""
    pass

def p_def(p):
    """def : keyword ID '=' '(' defbody ')'
    | keyword ID '=' '{' intrabody '}'
    | keyword ID '=' '[' interbody ']'
    | keyword ID '=' VALUE"""
    # p[0] : p[1] p[2] p[3] p[4] p[5] p[6]
    if p[1] == 'link_type':
        p[0] = LinkType(p[3],p[2],p[5])
        linktype_v.visit(p[0])
    elif p[1] == 'intra_node':
        p[0] = IntraNode(p[3],p[2],p[5])
        intra_v.visit(p[0])
    elif p[1] == 'inter_node':
        p[0] = InterNode(p[3],p[2],p[5])
        inter_v.visit(p[0])
    elif p[1] == 'define':
        p[0] = Definition(p[3],p[2],p[4])
        define_v.visit(p[0])

def p_value(p):
    """ VALUE : NUMBER
    """
    p[0] = p[1]

def p_keyword(p):
    """keyword : LINKTYPE
    | DEFINE
    | INTRA_NODE
    | INTER_NODE"""
    p[0] = p[1]

def p_defbody(p):
    """defbody : NUMBER ',' NUMBER"""
    # p[0] : p[1] p[2] p[3]
    p[0] = (p[1], p[3])

def p_intrabody(p):
    """intrabody : intra intrabodytail
    """
    p[0] = [p[1]] + p[2]

def p_intra_body_tail(p):
    """intrabodytail : empty
    | ',' intra intrabodytail"""
    p[0] = []
    if len(p) > 2:
        p[0] = [p[2]] + p[3]

def p_intra(p):
    """intra :  SWITCH CONSTRAINT '[' '(' tuple ')' WAYPOINT '(' ID ',' NUMBER ')' ']'
    """
    p[0] = {}
    p[0]['conn_type'] = p[1]
    p[0]['tuple'] = p[5]
    p[0]['linktype'] = p[9]
    p[0]['number'] = p[11]

def p_tuple(p):
    """tuple : NUMBER tupletail
    """
    p[0] = [p[1]] + p[2]

def p_tupletail(p):
    """tupletail : empty
    | ',' NUMBER tupletail
    """
    if len(p) > 2:
        p[0] = [p[2]] + p[3]
    else:
        p[0] = []

def p_interbody(p):
    """interbody : MATCH CONSTRAINT '(' tuple ')'
    """
    p[0] = {}
    p[0]['conn_type'] = p[1]
    p[0]['tuple'] = p[4]
    
def p_error(p):
    print("Syntax error at '%s'" % p.value)

# -------------------- test --------------------

# lexer = lex.lex() 

# parser = yacc.yacc()

# with open('./dsl/dsl/test.dsl', 'r') as file:
#     while True:
#         intent_line = file.readline()
#         if intent_line:
#             r = parser.parse(intent_line)
#             # print(r)
#         else:
#             break

#     print('linktype_v.lvalues', linktype_v.lvalues)
#     print('linktype_v.rvalues', linktype_v.rvalues)
#     print('define_v.lvalues', define_v.lvalues)
#     print('define_v.rvalues', define_v.rvalues)
#     print('intra_v.lvalues', intra_v.lvalues)
#     print('intra_v.rvalues', intra_v.rvalues)
#     print('inter_v.lvalues', inter_v.lvalues)
#     print('inter_v.rvalues', inter_v.rvalues)