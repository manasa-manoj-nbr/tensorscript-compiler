"""
Lexer for TensorScript DSL
Converts source code into tokens for parsing
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


class TokenType(Enum):
    # Literals
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    
    # Keywords
    MATMUL = auto()
    ADD = auto()
    RELU = auto()
    SOFTMAX = auto()
    TRANSPOSE = auto()
    
    # Operators
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    ASSIGN = auto()
    AT = auto()  # @ decorator
    
    # Special
    NEWLINE = auto()
    EOF = auto()
    

@dataclass
class Token:
    type: TokenType
    value: any
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


class Lexer:
    """Tokenizes TensorScript source code"""
    
    KEYWORDS = {
        'matmul': TokenType.MATMUL,
        'add': TokenType.ADD,
        'relu': TokenType.RELU,
        'softmax': TokenType.SOFTMAX,
        'transpose': TokenType.TRANSPOSE,
    }
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
    def error(self, msg: str):
        raise SyntaxError(f"Lexer error at {self.line}:{self.column}: {msg}")
    
    def peek(self, offset=0) -> Optional[str]:
        """Look ahead at character without consuming"""
        pos = self.pos + offset
        if pos < len(self.source):
            return self.source[pos]
        return None
    
    def advance(self) -> Optional[str]:
        """Consume and return current character"""
        if self.pos >= len(self.source):
            return None
        char = self.source[self.pos]
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
    
    def skip_whitespace(self):
        """Skip spaces and tabs (but not newlines)"""
        while self.peek() and self.peek() in ' \t':
            self.advance()
    
    def skip_comment(self):
        """Skip # comments to end of line"""
        if self.peek() == '#':
            while self.peek() and self.peek() != '\n':
                self.advance()
    
    def read_number(self) -> Token:
        """Read integer or float literal"""
        start_line = self.line
        start_col = self.column
        num_str = ''
        
        while self.peek() and (self.peek().isdigit() or self.peek() == '.'):
            num_str += self.advance()
        
        if '.' in num_str:
            value = float(num_str)
        else:
            value = int(num_str)
            
        return Token(TokenType.NUMBER, value, start_line, start_col)
    
    def read_identifier(self) -> Token:
        """Read identifier or keyword"""
        start_line = self.line
        start_col = self.column
        ident = ''
        
        while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
            ident += self.advance()
        
        # Check if it's a keyword
        token_type = self.KEYWORDS.get(ident, TokenType.IDENTIFIER)
        
        return Token(token_type, ident, start_line, start_col)
    
    def read_string(self) -> Token:
        """Read string literal"""
        start_line = self.line
        start_col = self.column
        
        quote = self.advance()  # consume opening quote
        string = ''
        
        while self.peek() and self.peek() != quote:
            if self.peek() == '\\':
                self.advance()
                escape = self.advance()
                if escape == 'n':
                    string += '\n'
                elif escape == 't':
                    string += '\t'
                elif escape == '\\':
                    string += '\\'
                elif escape == quote:
                    string += quote
            else:
                string += self.advance()
        
        if self.peek() != quote:
            self.error("Unterminated string")
        
        self.advance()  # consume closing quote
        
        return Token(TokenType.STRING, string, start_line, start_col)
    
    def tokenize(self) -> List[Token]:
        """Convert source code to token list"""
        while self.pos < len(self.source):
            self.skip_whitespace()
            self.skip_comment()
            
            if self.peek() is None:
                break
            
            char = self.peek()
            start_line = self.line
            start_col = self.column
            
            # Newline
            if char == '\n':
                self.advance()
                self.tokens.append(Token(TokenType.NEWLINE, '\n', start_line, start_col))
            
            # Numbers
            elif char.isdigit():
                self.tokens.append(self.read_number())
            
            # Identifiers and keywords
            elif char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
            
            # Strings
            elif char in '"\'':
                self.tokens.append(self.read_string())
            
            # Single-character tokens
            elif char == '(':
                self.advance()
                self.tokens.append(Token(TokenType.LPAREN, '(', start_line, start_col))
            elif char == ')':
                self.advance()
                self.tokens.append(Token(TokenType.RPAREN, ')', start_line, start_col))
            elif char == ',':
                self.advance()
                self.tokens.append(Token(TokenType.COMMA, ',', start_line, start_col))
            elif char == '=':
                self.advance()
                self.tokens.append(Token(TokenType.ASSIGN, '=', start_line, start_col))
            elif char == '@':
                self.advance()
                self.tokens.append(Token(TokenType.AT, '@', start_line, start_col))
            else:
                self.error(f"Unexpected character: {char!r}")
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        
        return self.tokens


def tokenize(source: str) -> List[Token]:
    """Convenience function to tokenize source code"""
    lexer = Lexer(source)
    return lexer.tokenize()


if __name__ == '__main__':
    # Test the lexer
    test_code = """
    # Matrix multiplication example
    result = matmul(A, B)
    activated = relu(result)
    """
    
    tokens = tokenize(test_code)
    for token in tokens:
        print(token)
