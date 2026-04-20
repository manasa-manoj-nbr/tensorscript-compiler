"""
Abstract Syntax Tree (AST) node definitions for TensorScript
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ASTNode:
    """Base class for all AST nodes"""
    line: int
    column: int


@dataclass
class Expression(ASTNode):
    """Base class for expressions"""
    pass


@dataclass
class Variable(Expression):
    """Variable reference: A, B, result"""
    name: str


@dataclass
class Number(Expression):
    """Numeric literal: 32, 1.5"""
    value: float


@dataclass
class FunctionCall(Expression):
    """Function call: matmul(A, B), relu(x)"""
    name: str
    args: List[Expression]
    kwargs: dict  # For named arguments like tile_size=32


@dataclass
class Assignment(ASTNode):
    """Assignment statement: result = matmul(A, B)"""
    target: str
    value: Expression


@dataclass
class Program(ASTNode):
    """Top-level program node"""
    statements: List[ASTNode]


def pretty_print(node: ASTNode, indent=0) -> str:
    """Pretty print AST for debugging"""
    prefix = "  " * indent
    
    if isinstance(node, Program):
        result = f"{prefix}Program:\n"
        for stmt in node.statements:
            result += pretty_print(stmt, indent + 1)
        return result
    
    elif isinstance(node, Assignment):
        result = f"{prefix}Assignment: {node.target} =\n"
        result += pretty_print(node.value, indent + 1)
        return result
    
    elif isinstance(node, FunctionCall):
        result = f"{prefix}FunctionCall: {node.name}\n"
        for arg in node.args:
            result += pretty_print(arg, indent + 1)
        if node.kwargs:
            result += f"{prefix}  kwargs: {node.kwargs}\n"
        return result
    
    elif isinstance(node, Variable):
        return f"{prefix}Variable: {node.name}\n"
    
    elif isinstance(node, Number):
        return f"{prefix}Number: {node.value}\n"
    
    else:
        return f"{prefix}{node.__class__.__name__}\n"
