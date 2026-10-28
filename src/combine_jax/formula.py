"""Allowlisted RooFormulaVar expressions lowered to a small immutable AST.

Workspace text is parsed, never passed to Python eval/exec or compiled as code.
"""

import ast
import math
import operator
import re

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfc

from .graph import UnsupportedModelError


FUNCTIONS = {
    "abs": (1, jnp.abs), "fabs": (1, jnp.abs), "sqrt": (1, jnp.sqrt),
    "exp": (1, jnp.exp), "log": (1, jnp.log), "log10": (1, jnp.log10),
    "sin": (1, jnp.sin), "cos": (1, jnp.cos), "tan": (1, jnp.tan),
    "tanh": (1, jnp.tanh), "erf": (1, erf), "erfc": (1, erfc),
    "pow": (2, jnp.power), "min": (2, jnp.minimum), "max": (2, jnp.maximum),
}
ALIASES = {"Abs": "abs", "Sqrt": "sqrt", "Exp": "exp", "Log": "log", "Log10": "log10",
           "Sin": "sin", "Cos": "cos", "Tan": "tan", "TanH": "tanh", "Erf": "erf", "Erfc": "erfc",
           "Power": "pow", "Min": "min", "Max": "max"}
BINARY = {ast.Add: ("+", operator.add), ast.Sub: ("-", operator.sub), ast.Mult: ("*", operator.mul),
          ast.Div: ("/", operator.truediv), ast.Pow: ("**", operator.pow)}
COMPARE = {ast.Lt: ("<", operator.lt), ast.LtE: ("<=", operator.le), ast.Gt: (">", operator.gt),
           ast.GtE: (">=", operator.ge), ast.Eq: ("==", operator.eq), ast.NotEq: ("!=", operator.ne)}


def rewrite_ternary(expression):
    """Translate C conditional syntax structurally before allowlisted AST parsing."""
    grouped, position = [], 0
    while position < len(expression):
        if expression[position] != "(":
            grouped.append(expression[position]); position += 1
            continue
        end, depth = position+1, 1
        while end < len(expression) and depth:
            depth += (expression[end] == "(") - (expression[end] == ")")
            end += 1
        if depth:
            raise UnsupportedModelError("Unbalanced formula parentheses")
        grouped.append("("+rewrite_ternary(expression[position+1:end-1])+")")
        position = end
    text = "".join(grouped)
    depth, commas, questions = 0, [], []
    for i, character in enumerate(text):
        depth += (character == "(") - (character == ")")
        if depth == 0 and character == ",": commas.append(i)
        if depth == 0 and character == "?": questions.append(i)
    if commas:
        edges = [-1, *commas, len(text)]
        return ",".join(rewrite_ternary(text[a+1:b]) for a, b in zip(edges[:-1], edges[1:], strict=True))
    if not questions:
        return text
    question, depth, pending = questions[0], 0, 0
    for i in range(question+1, len(text)):
        character = text[i]
        depth += (character == "(") - (character == ")")
        if depth:
            continue
        if character == "?": pending += 1
        if character == ":":
            if pending:
                pending -= 1
            else:
                return "(("+rewrite_ternary(text[question+1:i])+") if ("+text[:question]+") else ("+rewrite_ternary(text[i+1:])+"))"
    raise UnsupportedModelError("Formula conditional has no matching colon")


def parse_formula(expression, arguments):
    if len(expression) > 32768:
        raise UnsupportedModelError("Formula exceeds the expression size limit")
    if any(name.startswith("__argument_") for name in arguments):
        raise UnsupportedModelError("Formula dependency uses a reserved parser identifier")
    transformed = expression.replace("TMath::Pi()", repr(math.pi))
    # RooFit permits names such as 13TeV_lumi. Replace whole known tokens
    # before Python parses them as malformed numeric literals. Do not change
    # numeric literals (including scientific notation) or identifier substrings.
    for index, name in enumerate(arguments):
        if re.fullmatch(r"\d\w*", name):
            try:
                float(name)
            except ValueError:
                transformed = re.sub(r"(?<!\w)"+re.escape(name)+r"(?!\w)",
                                     f"__argument_{index}", transformed)
    for source, destination in ALIASES.items():
        transformed = transformed.replace("TMath::" + source + "(", destination + "(")
    transformed = re.sub(r"@(\d+)", r"__argument_\1", transformed)
    transformed = transformed.replace("^", "**").replace("&&", " and ").replace("||", " or ")
    transformed = re.sub(r"!(?!=)", " not ", transformed).strip()
    try:
        transformed = rewrite_ternary(transformed) if "?" in transformed else transformed
        tree = ast.parse(transformed, mode="eval").body
    except (SyntaxError, RecursionError) as exc:
        raise UnsupportedModelError(f"Unsupported formula syntax {expression!r}") from exc
    names = {name: i for i, name in enumerate(arguments)}

    def argument(index):
        if not 0 <= index < len(arguments):
            raise UnsupportedModelError(f"Formula argument index {index} is out of range")
        return ("argument", index)

    def lower(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            return ("constant", float(node.value))
        if isinstance(node, ast.Name):
            if node.id.startswith("__argument_") and node.id[11:].isdigit():
                return argument(int(node.id[11:]))
            if node.id in names:
                return argument(names[node.id])
            if node.id == "pi":
                return ("constant", math.pi)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "x":
            if isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
                return argument(node.slice.value)
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            return ("binary", BINARY[type(node.op)][0], lower(node.left), lower(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd, ast.Not):
            return ("unary", {ast.USub: "-", ast.UAdd: "+", ast.Not: "not"}[type(node.op)], lower(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
            name = node.func.id
            if node.keywords or len(node.args) != FUNCTIONS[name][0]:
                raise UnsupportedModelError(f"Formula function {name} has unsupported arguments")
            return ("call", name, tuple(lower(arg) for arg in node.args))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in COMPARE:
            return ("compare", COMPARE[type(node.ops[0])][0], lower(node.left), lower(node.comparators[0]))
        if isinstance(node, ast.BoolOp) and type(node.op) in (ast.And, ast.Or):
            return ("boolean", "and" if isinstance(node.op, ast.And) else "or", tuple(lower(v) for v in node.values))
        if isinstance(node, ast.IfExp):
            return ("conditional", lower(node.test), lower(node.body), lower(node.orelse))
        raise UnsupportedModelError(f"Unsupported formula element {ast.dump(node, include_attributes=False)}")

    return lower(tree)


def evaluate_formula(tree, values):
    kind = tree[0]
    if kind == "argument":
        return values[tree[1]]
    if kind == "constant":
        return jnp.asarray(tree[1])
    if kind in ("binary", "compare"):
        functions = BINARY if kind == "binary" else COMPARE
        operation = next(fn for name, fn in functions.values() if name == tree[1])
        return operation(evaluate_formula(tree[2], values), evaluate_formula(tree[3], values))
    if kind == "unary":
        operation = {"-": operator.neg, "+": operator.pos, "not": jnp.logical_not}[tree[1]]
        return operation(evaluate_formula(tree[2], values))
    if kind == "boolean":
        operation = jnp.logical_and if tree[1] == "and" else jnp.logical_or
        result = evaluate_formula(tree[2][0], values)
        for operand in tree[2][1:]:
            result = operation(result, evaluate_formula(operand, values))
        return result
    if kind == "call":
        return FUNCTIONS[tree[1]][1](*(evaluate_formula(arg, values) for arg in tree[2]))
    if kind == "conditional":
        def branch(t, v):
            shape = jnp.broadcast_shapes(*(jnp.shape(x) for x in v))
            return jnp.broadcast_to(jnp.asarray(evaluate_formula(t, v), dtype=jnp.float64), shape)
        def scalar(v):
            return jax.lax.cond(jnp.asarray(evaluate_formula(tree[1], v), dtype=bool),
                                lambda x: branch(tree[2], x), lambda x: branch(tree[3], x), v)
        condition = evaluate_formula(tree[1], values)
        if jnp.ndim(condition) == 0:
            return scalar(tuple(values))
        # A sequential mapped conditional preserves short-circuit derivatives
        # even when an inactive branch contains sqrt/log of an invalid value.
        # vmap(cond) would lower to select and can reintroduce NaN gradients.
        broadcast = jnp.broadcast_arrays(*values)
        shape = broadcast[0].shape
        return jax.lax.map(scalar, tuple(v.reshape(-1) for v in broadcast)).reshape(shape)
    raise AssertionError(f"Invalid formula IR {kind}")


def extract_formula(root):
    arguments = tuple(str(item.GetName()) for item in root.dependents())
    expression = str(root.expression())
    return arguments, {"expression": expression, "ast": parse_formula(expression, arguments)}


def formula_value(node, context):
    return jnp.asarray(evaluate_formula(node.attrs["ast"], context.values(node.inputs)), dtype=jnp.float64)
