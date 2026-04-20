# Project Structure

```
tensorscript-compiler/
│
├── Core Compiler Components
│   ├── lexer.py              # Tokenization (source → tokens)
│   ├── ast_nodes.py          # AST node definitions
│   ├── parser.py             # Parsing (tokens → AST)
│   ├── ir.py                 # IR data structures (SSA form)
│   ├── ir_generator.py       # IR generation (AST → IR)
│   ├── cuda_generator.py     # CUDA code generation (IR → CUDA)
│   └── ptx_generator.py      # PTX assembly generation (IR → PTX)
│
├── Main Entry Points
│   ├── compiler.py           # Main compiler driver & CLI
│   └── demo.py               # Interactive demonstration
│
├── Examples & Tests
│   ├── examples.ts           # Example TensorScript programs
│   ├── neural_network.ts     # Neural network example
│   └── test_compiler.py      # Unit and integration tests
│
├── Documentation
│   ├── README.md             # Main documentation
│   └── ARCHITECTURE.md       # This file
│
└── Configuration
    ├── requirements.txt      # Python dependencies (none!)
    ├── .gitignore           # Git ignore rules
```

## Data Flow

```
┌─────────────────┐
│  TensorScript   │
│     Source      │
│  (examples.ts)  │
└────────┬────────┘
         │
         ▼
    ┌────────┐
    │ Lexer  │ ──> Token stream
    └────┬───┘
         │
         ▼
    ┌────────┐
    │ Parser │ ──> Abstract Syntax Tree (AST)
    └────┬───┘
         │
         ▼
  ┌──────────────┐
  │ IR Generator │ ──> Intermediate Representation (SSA)
  └──────┬───────┘
         │
         ├────────────┬────────────┐
         ▼            ▼            ▼
   ┌─────────┐  ┌─────────┐  ┌──────┐
   │  CUDA   │  │   PTX   │  │  IR  │
   │Generator│  │Generator│  │(text)│
   └────┬────┘  └────┬────┘  └──────┘
        │            │
        ▼            ▼
    output.cu    output.ptx
```

## Module Dependencies

```
compiler.py
    ↓
    ├── lexer.py
    ├── parser.py (→ ast_nodes.py, lexer.py)
    ├── ir_generator.py (→ ast_nodes.py, ir.py)
    └── Code Generators
        ├── cuda_generator.py (→ ir.py)
        └── ptx_generator.py (→ ir.py)
```

## Key Design Decisions

### 1. No External Dependencies
- Uses only Python standard library
- Easy to install and understand
- No version conflicts

### 2. SSA-Form IR
- Simplifies optimization passes
- Clear data flow
- Industry standard (LLVM, GCC)

### 3. Separate Code Generators
- CUDA and PTX are independent
- Easy to add new targets
- Clean separation of concerns

### 4. Immutable AST
- AST nodes are read-only after creation
- Safer transformations
- Easier debugging

### 5. Multi-Phase Compilation
- Each phase is testable independently
- Clear interfaces between phases
- Educational structure

## Extension Points

### Adding New Operations

1. **Lexer**: Add keyword to `KEYWORDS` dict
2. **AST**: Add node type if needed (usually reuse FunctionCall)
3. **IR**: Add operation type to `IROpType` enum
4. **Code Gen**: Implement in CUDA/PTX generators

### Adding Optimizations

Modify `ir_generator.py` to:
- Implement constant folding
- Do operation fusion
- Eliminate dead code
- Perform common subexpression elimination

### Adding New Targets

Create new generator (e.g., `llvm_generator.py`):
- Inherit from or mimic existing generators
- Traverse IR and emit target code
- Add to `compiler.py` output format options

## Performance Characteristics

| Phase | Time Complexity | Notes |
|-------|----------------|-------|
| Lexer | O(n) | Linear in source size |
| Parser | O(n) | Recursive descent |
| IR Gen | O(n) | Linear in AST size |
| Code Gen | O(n) | Linear in IR size |

**Total**: O(n) where n is source code size

Memory usage is proportional to program size (AST + IR both in memory).

## Testing Strategy

### Unit Tests
- Lexer: Token generation
- Parser: AST construction
- IR Generator: Correctness of IR

### Integration Tests
- End-to-end compilation
- Multi-operation programs
- Optimization validation

### Manual Testing
- Interactive demo
- Example programs
- Visual inspection of generated code

## Code Quality

### Style
- PEP 8 compliant
- Type hints in key locations
- Docstrings for all public functions

### Error Handling
- Meaningful error messages
- Line/column information for syntax errors
- Graceful degradation

### Documentation
- Inline comments for complex logic
- README for overview
- Docstrings for API documentation
- Separate guides for different audiences
