"""
Parser for TensorScript DSL
Converts tokens into Abstract Syntax Tree (AST)
"""

from typing import List, Optional
from lexer import Token, TokenType
from ast_nodes import *


class Parser:
    """Recursive descent parser for TensorScript"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        
    def error(self, msg: str):
        token = self.current()
        raise SyntaxError(f"Parse error at {token.line}:{token.column}: {msg}")
    
    def current(self) -> Token:
        """Get current token without advancing"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # Return EOF
    
    def peek(self, offset=1) -> Token:
        """Look ahead at future token"""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]
    
    def advance(self) -> Token:
        """Consume and return current token"""
        token = self.current()
        if token.type != TokenType.EOF:
            self.pos += 1
        return token
    
    def expect(self, token_type: TokenType) -> Token:
        """Consume token of expected type or error"""
        token = self.current()
        if token.type != token_type:
            self.error(f"Expected {token_type.name}, got {token.type.name}")
        return self.advance()
    
    def skip_newlines(self):
        """Skip any newline tokens"""
        while self.current().type == TokenType.NEWLINE:
            self.advance()
    
    def parse(self) -> Program:
        """Parse entire program"""
        statements = []
        
        self.skip_newlines()
        
        while self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()
        
        return Program(statements=statements, line=1, column=1)
    
    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement"""
        token = self.current()
        
        # Assignment: result = matmul(A, B)
        if token.type == TokenType.IDENTIFIER and self.peek().type == TokenType.ASSIGN:
            return self.parse_assignment()
        
        # Expression statement (just a function call)
        else:
            expr = self.parse_expression()
            return expr
    
    def parse_assignment(self) -> Assignment:
        """Parse assignment: target = expression"""
        target_token = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()
        
        return Assignment(
            target=target_token.value,
            value=value,
            line=target_token.line,
            column=target_token.column
        )
    
    def parse_expression(self) -> Expression:
        """Parse an expression"""
        token = self.current()
        
        # Function call
        if token.type in [TokenType.MATMUL, TokenType.ADD, TokenType.RELU, 
                          TokenType.SOFTMAX, TokenType.TRANSPOSE]:
            return self.parse_function_call()
        
        # Variable
        elif token.type == TokenType.IDENTIFIER:
            var_token = self.advance()
            return Variable(name=var_token.value, line=var_token.line, column=var_token.column)
        
        # Number
        elif token.type == TokenType.NUMBER:
            num_token = self.advance()
            return Number(value=num_token.value, line=num_token.line, column=num_token.column)
        
        else:
            self.error(f"Unexpected token in expression: {token.type.name}")
    
    def parse_function_call(self) -> FunctionCall:
        """Parse function call: matmul(A, B) or add(x, y, tile_size=32)"""
        func_token = self.advance()
        func_name = func_token.value
        
        self.expect(TokenType.LPAREN)
        
        args = []
        kwargs = {}
        
        # Parse arguments
        if self.current().type != TokenType.RPAREN:
            while True:
                # Check for keyword argument (name=value)
                if (self.current().type == TokenType.IDENTIFIER and 
                    self.peek().type == TokenType.ASSIGN):
                    key = self.advance().value
                    self.expect(TokenType.ASSIGN)
                    value_token = self.expect(TokenType.NUMBER)
                    kwargs[key] = value_token.value
                else:
                    # Positional argument
                    args.append(self.parse_expression())
                
                # Check for more arguments
                if self.current().type == TokenType.COMMA:
                    self.advance()
                else:
                    break
        
        self.expect(TokenType.RPAREN)
        
        return FunctionCall(
            name=func_name,
            args=args,
            kwargs=kwargs,
            line=func_token.line,
            column=func_token.column
        )


def parse(tokens: List[Token]) -> Program:
    """Convenience function to parse tokens into AST"""
    parser = Parser(tokens)
    return parser.parse()


if __name__ == '__main__':
    from lexer import tokenize
    
    # Test the parser
    test_code = """
    result = matmul(A, B)
    activated = relu(result)
    output = softmax(activated)
    """
    
    tokens = tokenize(test_code)
    ast = parse(tokens)
    print(pretty_print(ast))
