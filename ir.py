"""
Intermediate Representation (IR) for TensorScript
A simplified SSA-form IR for tensor operations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum, auto


class IROpType(Enum):
    """Types of IR operations"""
    MATMUL = auto()
    ADD = auto()
    RELU = auto()
    SOFTMAX = auto()
    TRANSPOSE = auto()
    LOAD = auto()  # Load from variable
    STORE = auto()  # Store to variable


@dataclass
class IRValue:
    """A value in the IR (SSA form)"""
    name: str  # e.g., %0, %1, %result
    type: str  # e.g., "tensor<f32>"
    
    def __repr__(self):
        return self.name


@dataclass
class IROperation:
    """Single operation in IR"""
    op_type: IROpType
    result: IRValue
    operands: List[IRValue]
    attributes: Dict[str, any] = field(default_factory=dict)
    
    def __repr__(self):
        op_name = self.op_type.name.lower()
        operand_str = ", ".join(str(op) for op in self.operands)
        
        if self.attributes:
            attr_str = ", ".join(f"{k}={v}" for k, v in self.attributes.items())
            return f"{self.result} = {op_name}({operand_str}, {attr_str})"
        else:
            return f"{self.result} = {op_name}({operand_str})"


@dataclass
class IRBasicBlock:
    """Basic block of IR operations"""
    name: str
    operations: List[IROperation] = field(default_factory=list)
    
    def add_op(self, op: IROperation):
        self.operations.append(op)
    
    def __repr__(self):
        result = f"Block {self.name}:\n"
        for op in self.operations:
            result += f"  {op}\n"
        return result


@dataclass
class IRFunction:
    """IR function (represents a kernel)"""
    name: str
    inputs: List[IRValue]
    outputs: List[IRValue]
    blocks: List[IRBasicBlock] = field(default_factory=list)
    
    def add_block(self, block: IRBasicBlock):
        self.blocks.append(block)
    
    def __repr__(self):
        result = f"Function {self.name}:\n"
        result += f"  Inputs: {', '.join(str(i) for i in self.inputs)}\n"
        result += f"  Outputs: {', '.join(str(o) for o in self.outputs)}\n\n"
        for block in self.blocks:
            result += str(block) + "\n"
        return result


@dataclass
class IRModule:
    """Top-level IR module"""
    functions: List[IRFunction] = field(default_factory=list)
    
    def add_function(self, func: IRFunction):
        self.functions.append(func)
    
    def __repr__(self):
        result = "=== IR Module ===\n\n"
        for func in self.functions:
            result += str(func) + "\n"
        return result


class IRBuilder:
    """Helper class to build IR from AST"""
    
    def __init__(self):
        self.temp_counter = 0
        self.symbol_table: Dict[str, IRValue] = {}
    
    def new_temp(self, prefix="t") -> IRValue:
        """Generate new temporary value"""
        name = f"%{prefix}{self.temp_counter}"
        self.temp_counter += 1
        return IRValue(name=name, type="tensor<f32>")
    
    def get_or_create_value(self, name: str) -> IRValue:
        """Get existing value or create new one"""
        if name not in self.symbol_table:
            self.symbol_table[name] = IRValue(name=f"%{name}", type="tensor<f32>")
        return self.symbol_table[name]


def print_ir(module: IRModule) -> str:
    """Pretty print IR module"""
    return str(module)
