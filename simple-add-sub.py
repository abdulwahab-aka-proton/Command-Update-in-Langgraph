from langgraph.graph import StateGraph, END
from langgraph.types import Command
from typing import TypedDict

class AgentState(TypedDict):
    num1: int
    num2: int
    num3: int
    num4: int
    operation1: str
    operation2: str
    answer1: int
    answer2: int

def add_node(state: AgentState) -> Command:
    """This node adds num1 and num2"""
    result = state['num1'] + state['num2']
    return Command(
        goto="router2",
        update={"answer1": result}
    )

def add_node2(state: AgentState) -> Command:
    """This node adds num3 and num4"""
    result = state['num3'] + state['num4']
    return Command(
        goto=END,
        update={"answer2": result}
    )

def sub_node(state: AgentState) -> Command:
    """This node subtracts num2 from num1"""
    result = state['num1'] - state['num2']
    return Command(
        goto="router2",
        update={"answer1": result}
    )

def sub_node2(state: AgentState) -> Command:
    """This node subtracts num4 from num3"""
    result = state['num3'] - state['num4']
    return Command(
        goto=END,
        update={"answer2": result}
    )

def router_node1(state: AgentState) -> Command:
    """This node decides next node in graph based on operation1"""
    if state['operation1'] == "+":
        return Command(goto="add1")
    elif state['operation1'] == "-":
        return Command(goto="sub1")

def router_node2(state: AgentState) -> Command:
    """This node decides next node in graph based on operation2"""
    if state['operation2'] == "+":
        return Command(goto="add2")
    elif state['operation2'] == "-":
        return Command(goto="sub2")

workflow = StateGraph(AgentState)
workflow.add_node("router1", router_node1)
workflow.add_node("router2", router_node2)
workflow.add_node("add1", add_node)
workflow.add_node("add2", add_node2)
workflow.add_node("sub1", sub_node)
workflow.add_node("sub2", sub_node2)
workflow.set_entry_point("router1")
app = workflow.compile()

result = app.invoke({
    "num1": 5,
    "num2": 10,
    "operation1": "+",
    "num3": 10,
    "num4": 6,
    "operation2": "-"
})

print(f"Your Answer#1 is: {result['answer1']}\nYour Answer#2 is: {result['answer2']}")
