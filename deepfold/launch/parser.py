#!/usr/bin/env python3


import sys

import ply.lex as lex
import ply.yacc as yacc

RE_ID = r"[a-zA-Z_][a-zA-Z_0-9]*"
RE_INT = r"[1-9][0-9]*"

TOKENS = [
    # Initial state tokens
    "BEGIN_ARRAY",
    "BEGIN_OBJECT",
    "END_ARRAY",
    "END_OBJECT",
    "NAME_SEPARATOR",
    "VALUE_SEPARATOR",
    "QUOTATION_MARK",
    "FALSE",
    "TRUE",
    "NULL",
    "DECIMAL_POINT",
    "DIGITS",
    "E",
    "MINUS",
    "PLUS",
    "ZERO",
    # String state tokens
    "UNESCAPED",
    "ESCAPE",
    # Escaped state tokens
    "REVERSE_SOLIDUS",
    "SOLIDUS",
    "BACKSPACE_CHAR",
    "FORM_FEED_CHAR",
    "LINE_FEED_CHAR",
    "CARRIAGE_RETURN_CHAR",
    "TAB_CHAR",
    "UNICODE_HEX",
    # Graph
    "BEGIN_EDGE_LIST",  # [[
    "END_EDGE_LIST",  # ]]
    "EDGE_SEP",  # -
    # Predict
    "STOI_SEP",
    "NUM_SYM_SEP",
    # "STOI_NUM",
    # Identifier
    "ID",
    "SEMICOLON",
    "NEWLINE",
]

RESERVED = {
    # Types
    "model": "MODEL",
    "entity": "ENTITY",
    "predict": "PREDICT",
    # Pred
    "using": "USING",
    "stoi": "STOI",
    "pair": "PAIR",
    "sample": "SAMPLE",
    # Varibles
    "graph": "GRAPH",
    "let": "LET",
    "in": "IN",
}

TOKENS += RESERVED.values()


class Lexer:

    def __init__(self, debug=False, **kwargs):
        self.lexer = lex.lex(
            module=self,
            debug=debug,
            **kwargs,
        )

    tokens = TOKENS

    states = (
        ("iterable", "inclusive"),
        ("string", "exclusive"),
        ("escaped", "exclusive"),
        ("graph", "inclusive"),
        ("stoi", "exclusive"),
    )

    def t_ANY_error(self, t):
        last_cr = self.lexer.lexdata.rfind("\n", 0, t.lexpos)
        if last_cr < 0:
            last_cr = 0
        column = t.lexpos - last_cr + 1
        print(f"Illegal character '{t.value[0]}' at line {t.lineno} pos {column}")
        t.lexer.skip(1)

    t_SEMICOLON = r"\x3B"

    # Count newline
    def t_NEWLINE(self, t):
        r"\n+"
        t.lexer.lineno += len(t.value)
        return t

    # Skips over '\s', '\t', and '\r' characters in the default state
    t_ignore = "\x20\x09\x0D"

    @lex.TOKEN(RE_ID)
    def t_ID(self, t):

        if t.value in RESERVED:
            t.type = RESERVED[t.value]

        if t.value == "graph":
            t.lexer.push_state("graph")

        if t.value == "stoi":
            t.lexer.push_state("stoi")

        return t

    # Iterable

    def t_iterable_NEWLINE(self, t):
        r"\n+"
        t.lexer.lineno += len(t.value)

    # t_BEGIN_ARRAY = r"\x5B"  # '['
    # t_BEGIN_OBJECT = r"\x7B"  # '{'

    def t_BEGIN_ARRAY(self, t):
        r"\x5B"
        t.lexer.push_state("iterable")
        return t

    def t_BEGIN_OBJECT(self, t):
        r"\x7B"
        t.lexer.push_state("iterable")
        return t

    def t_iterable_END_ARRAY(self, t):
        r"\x5D"
        t.lexer.pop_state()
        return t

    def t_iterable_END_OBJECT(self, t):
        r"\x7D"
        t.lexer.pop_state()
        return t

    # t_END_ARRAY = r"\x5D"  # ']'
    # t_END_OBJECT = r"\x7D"  # '}'

    # Default state tokens

    t_NAME_SEPARATOR = r"\x3A"  # ':'
    t_VALUE_SEPARATOR = r"\x2C"  # ','
    t_FALSE = r"\x66\x61\x6c\x73\x65"  # 'false'
    t_TRUE = r"\x74\x72\x75\x65"  # 'true'
    t_NULL = r"\x6e\x75\x6c\x6c"  # 'null'
    t_DECIMAL_POINT = r"\x2E"  # '.'
    t_DIGITS = r"[\x30-\x39]+"  # '0'..'9'
    t_E = r"[\x45\x65]"  # 'e' or 'E'
    t_MINUS = r"\x2D"  # '-'
    t_PLUS = r"\x2B"  # '+'
    t_ZERO = r"\x30"  # '0'

    def t_ignore_COMMENT(self, t):
        r"\#.*"

    # Graph

    @lex.TOKEN(RE_ID)
    def t_graph_ID(self, t):
        if t.value in RESERVED:
            t.type = RESERVED[t.value]
        return t

    def t_graph_BEGIN_EDGE_LIST(self, t):
        r"\x5B\x5B"  # [[
        return t

    def t_graph_EDGE_SEP(self, t):
        r"\x2D"
        return t

    t_graph_VALUE_SEPARATOR = "\x2C"  # ','
    t_graph_ignore = "\x20\x09\x0D"

    # Ignore \n
    def t_graph_NEWLINE(self, t):
        r"\n+"
        t.lexer.lineno += len(t.value)

    def t_graph_END_EDGE_LIST(self, t):
        r"\x5D\x5D"  # ]]
        t.lexer.pop_state()
        return t

    # Enters the string state on an opening quotation mark
    def t_QUOTATION_MARK(self, t):
        r"\x22"  # '"'
        t.lexer.push_state("string")
        return t

    # Don't skip over any tokens inside the string state
    t_string_ignore = ""

    # TODO(dewitt): Verify that this matches the correct range, the spec
    # says '%x5D-10FFFF' but most pythons by default will not handle that
    def t_string_UNESCAPED(self, t):
        r"[\x20-\x21,\x23-\x5B,\x5D-\xFF]+"
        t.value = str(t.value)
        return t

    # Exits the string state on an unescaped closing quotation mark
    def t_string_QUOTATION_MARK(self, t):
        r"\x22"  # '"'
        t.lexer.pop_state()
        return t

    # Enter the escaped state on a '\' character
    def t_string_ESCAPE(self, t):
        r"\x5C"  # '\'
        t.lexer.push_state("escaped")
        return t

    # Don't skip over any tokens inside the escaped state
    t_escaped_ignore = ""

    def t_escaped_QUOTATION_MARK(self, t):
        r"\x22"  # '"'
        t.lexer.pop_state()
        return t

    def t_escaped_REVERSE_SOLIDUS(self, t):
        r"\x5C"  # '\'
        t.lexer.pop_state()
        return t

    def t_escaped_SOLIDUS(self, t):
        r"\x2F"  # '/'
        t.lexer.pop_state()
        return t

    def t_escaped_BACKSPACE_CHAR(self, t):
        r"\x62"  # 'b'
        t.lexer.pop_state()
        t.value = chr(0x0008)
        return t

    def t_escaped_FORM_FEED_CHAR(self, t):
        r"\x66"  # 'f'
        t.lexer.pop_state()
        t.value = chr(0x000C)
        return t

    def t_escaped_CARRIAGE_RETURN_CHAR(self, t):
        r"\x72"  # 'r'
        t.lexer.pop_state()
        t.value = chr(0x000D)
        return t

    def t_escaped_LINE_FEED_CHAR(self, t):
        r"\x6E"  # 'n'
        t.lexer.pop_state()
        t.value = chr(0x000A)
        return t

    def t_escaped_TAB_CHAR(self, t):
        r"\x74"  # 't'
        t.lexer.pop_state()
        t.value = chr(0x0009)
        return t

    def t_escaped_UNICODE_HEX(self, t):
        r"\x75[\x30-\x39,\x41-\x46,\x61-\x66]{4}"  # 'uXXXX'
        t.lexer.pop_state()
        return t

    # Stoichem

    @lex.TOKEN(RE_ID)
    def t_stoi_ID(self, t):
        if t.value in RESERVED:
            t.type = RESERVED[t.value]
        if t.value == "using":
            t.lexer.pop_state()
        return t

    def t_stoi_NUM_SYM_SEP(self, t):
        r"\x3A"
        return t

    def t_stoi_STOI_SEP(self, t):
        r"\x2F"
        return t

    @lex.TOKEN(RE_INT)
    def t_stoi_DIGITS(self, t):
        return t

    t_stoi_ignore = "\x20\x09\x0D"

    # Tokenizer

    def tokenize(self, data, *args, **kwargs):
        """Invoke the lexer on an input string an return the list of tokens.

        This is relatively inefficient and should only be used for
        testing/debugging as it slurps up all tokens into one list.

        Args:
          data: The input to be tokenized.
        Returns:
          A list of LexTokens
        """
        self.lexer.input(data)
        tokens = list()
        while True:
            token = self.lexer.token()
            if not token:
                break
            tokens.append(token)
        return tokens


class Parser:

    def __init__(self, lexer=None, debug=False, **kwargs):
        if lexer is not None:
            if isinstance(lexer, Lexer):
                self.lexer = lexer.lexer
            else:
                # Assume that the lexer is a lex instance or similar
                self.lexer = lexer
        else:
            self.lexer = Lexer(debug=debug).lexer
        self.parser = yacc.yacc(
            module=self,
            debug=debug,
            write_tables=False,
            **kwargs,
        )
        self.debug = debug

    tokens = TOKENS

    # Define the parser
    def p_script(self, p):
        """
        script : statements
        """
        p[0] = p[1:]

    def p_statements(self, p):
        """
        statements : statement
                   | statements NEWLINE statement
                   | statements SEMICOLON statement
        """
        if len(p) == 2:
            p[0] = [p[1]]
        elif len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]
        elif len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]

    def p_statement(self, p):
        """
        statement : command
        """
        p[0] = p[1]

    def p_command(self, p):
        """
        command :
                | model
                | entity
                | predict
                | graph
                | def
        """
        if len(p) == 1:
            p[0] = None
        else:
            p[0] = p[1]

    def p_id(self, p):
        """id : ID"""
        p[0] = f"${p[1]}"

    def p_variable_let(self, p):
        """
        def : LET id object
        """
        p[0] = {"command": "variable", "id": p[2], "value": p[3]}

    def p_model_definition(self, p):
        """
        model : MODEL id object
        """
        p[0] = {"command": "model", "id": p[2], "models": p[3]}

    def p_predict(self, p):
        """
        predict : PREDICT string STOI stoi_list USING id
                | PREDICT string STOI stoi_list USING id pred_options
                | PREDICT id STOI stoi_list USING id
                | PREDICT id STOI stoi_list USING id pred_options
        """
        if len(p) == 7:
            p[0] = {"command": "predict", "name": p[2], "stoi": p[4], "model": p[6], "options": []}
        elif len(p) == 8:
            p[0] = {"command": "predict", "name": p[2], "stoi": p[4], "model": p[6], "options": p[7]}

    def p_pred_opts(self, p):
        """
        pred_options : pred_option
                     | pred_options pred_option
        """
        if len(p) == 1:
            p[0] = []
        elif len(p) == 2:
            p[0] = [p[1]]
        elif len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]

    def p_pred_opt(self, p):
        """
        pred_option : PAIR string object
                    | SAMPLE string object
                    | IN object
        """
        if len(p) == 4:
            p[0] = {"command": p[1], "mode": p[2], "options": p[3]}
        elif len(p) == 3:
            p[0] = {"command": p[1], "mode": "path", "options": p[2]}

    def p_object(self, p):
        """
        object :
               | value
        """
        if len(p) == 1:
            p[0] = None
        else:
            p[0] = p[1]

    def p_entity(self, p):
        """
        entity : ENTITY id dict
               | ENTITY id string
        """
        p[0] = {"command": "entity", "id": p[2], "options": p[3]}

    def p_stoi_list(self, p):
        """
        stoi_list :
                  | stoi_list stoi_entry STOI_SEP
                  | stoi_list stoi_entry
        """
        if len(p) == 1:
            p[0] = list()
        else:
            p[1].append(p[2])
            p[0] = p[1]

    def p_stoi_entry(self, p):
        """
        stoi_entry : id NUM_SYM_SEP integer
        """
        p[0] = (p[1], p[3])

    def p_graph(self, p):
        """
        graph : GRAPH id edge_list
        """
        p[0] = {"command": "graph", "id": p[2], "edge_list": p[3]}

    def p_begin_edge(self, p):
        """
        begin_edge : BEGIN_EDGE_LIST
        """
        p[0] = None

    def p_end_edge(self, p):
        """
        end_edge : END_EDGE_LIST
        """
        p[0] = None

    def p_edge_list(self, p):
        """
        edge_list : begin_edge edges end_edge
        """
        p[0] = list(p[2])

    def p_edges(self, p):
        """
        edges :
              | edges edge value_separator
              | edges edge
        """
        if len(p) == 1:
            p[0] = list()
        else:
            p[1].append(p[2])
            p[0] = p[1]

    def p_edge(self, p):
        """
        edge : id EDGE_SEP id
        """
        p[0] = (p[1], p[3])

    # JSON parser

    def p_value(self, p):
        """
        value : dict
              | array
              | number
              | string
              | id
        """
        p[0] = p[1]

    def p_value_false(self, p):
        """value : FALSE"""
        p[0] = False

    def p_value_true(self, p):
        """value : TRUE"""
        p[0] = True

    def p_value_null(self, p):
        """value : NULL"""
        p[0] = None

    def p_begin_dict(self, p):
        """
        begin_dict : BEGIN_OBJECT
        """
        p[0] = None

    def p_end_dict(self, p):
        """
        end_dict : END_OBJECT
        """
        p[0] = None

    def p_dict(self, p):
        """dict : begin_dict members end_dict"""
        p[0] = dict(p[2])

    def p_value_seperator(self, p):
        """
        value_separator : VALUE_SEPARATOR
        """
        p[0] = None

    def p_members(self, p):
        """
        members :
                | members member value_separator
                | members member
        """
        if len(p) == 1:
            p[0] = list()
        else:
            p[1].append(p[2])
            p[0] = p[1]

    def p_member(self, p):
        """member : string NAME_SEPARATOR value"""
        p[0] = (p[1], p[3])

    def p_values(self, p):
        """
        values :
               | values value value_separator
               | values value
        """
        if len(p) == 1:
            p[0] = list()
        else:
            p[1].append(p[2])
            p[0] = p[1]

    def p_begin_array(self, p):
        """
        begin_array : BEGIN_ARRAY
        """
        p[0] = None

    def p_end_array(self, p):
        """
        end_array : END_ARRAY
        """
        p[0] = None

    def p_array(self, p):
        """array : begin_array values end_array"""
        p[0] = p[2]

    def p_number_positive(self, p):
        """
        number : integer
               | float
        """
        p[0] = p[1]

    def p_number_negative(self, p):
        """number : MINUS integer
        | MINUS float"""
        p[0] = -p[2]

    def p_integer(self, p):
        """integer : int"""
        p[0] = p[1]

    def p_integer_exp(self, p):
        """integer : int exp"""
        p[0] = p[1] * (10 ** p[2])

    def p_number_float(self, p):
        """float : int frac"""
        p[0] = p[1] + p[2]

    def p_number_float_exp(self, p):
        """float : int frac exp"""
        p[0] = (p[1] + p[2]) * (10 ** p[3])

    def p_exp_negative(self, p):
        """exp : E MINUS DIGITS"""
        p[0] = -int(p[3])

    def p_exp(self, p):
        """exp : E DIGITS"""
        p[0] = int(p[2])

    def p_exp_positive(self, p):
        """exp : E PLUS DIGITS"""
        p[0] = int(p[3])

    def p_frac(self, p):
        """frac : DECIMAL_POINT DIGITS"""
        p[0] = float("." + p[2])

    def p_int_zero(self, p):
        """int : ZERO"""
        p[0] = int(0)

    def p_int_non_zero(self, p):
        """int : DIGITS"""
        if p[1].startswith("0"):
            raise SyntaxError("Leading zeroes are not allowed.")
        p[0] = int(p[1])

    def p_string(self, p):
        """string : QUOTATION_MARK chars QUOTATION_MARK"""
        p[0] = p[2]

    def p_chars(self, p):
        """
        chars :
              | chars char
        """
        if len(p) == 1:
            p[0] = str()
        else:
            p[0] = p[1] + p[2]

    def p_char(self, p):
        """
        char : UNESCAPED
        | ESCAPE QUOTATION_MARK
        | ESCAPE REVERSE_SOLIDUS
        | ESCAPE SOLIDUS
        | ESCAPE BACKSPACE_CHAR
        | ESCAPE FORM_FEED_CHAR
        | ESCAPE LINE_FEED_CHAR
        | ESCAPE CARRIAGE_RETURN_CHAR
        | ESCAPE TAB_CHAR
        """
        # Because the subscript [-1] has special meaning for YaccProduction
        # slices we use [len(p) - 1] to always take the last value.
        p[0] = p[len(p) - 1]

    def p_char_unicode_hex(self, p):
        """char : ESCAPE UNICODE_HEX"""
        # This looks more complicated than it is.  The escaped string is of
        # the form \uXXXX and is assigned to p[2].  We take the trailing
        # XXXX string via p[2][1:], parse it as a radix 16 (hex) integer,
        # and convert that to the corresponding unicode character.
        p[0] = chr(int(p[2][1:], 16))

    def p_error(self, p):
        print(f"Syntax error at '{p}'")

    # Invoke the parser
    def parse(self, data, lexer=None, *args, **kwargs):
        if lexer is None:
            lexer = self.lexer

        if self.debug:
            lexer.input(data)
            print("=== TOKEN BEGIN ===")
            while True:
                tok = lexer.token()
                if not tok:
                    break
                print(tok)
            print("==== TOKEN END ====")

        lines = self.parser.parse(data, lexer=lexer, *args, **kwargs)
        if lines is not None:
            lines = [x for x in lines[0] if x is not None]
        else:
            lines = []

        return lines


# Maintain a reusable parser instance
parser = None
