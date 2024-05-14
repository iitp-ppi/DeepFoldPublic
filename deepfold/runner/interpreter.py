import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import networkx as nx

from deepfold.runner.parser import Parser

logger = logging.getLogger(__name__)


class Interpreter:

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

        # Logging
        global logger
        self.logger = logger
        if self.debug:
            logger.setLevel(logging.DEBUG)

        # Variables
        self.vars = dict()
        self.commands = []

        # Parser
        self.parser = Parser(debug=self.debug)

    def get(self, name: str) -> Any:
        try:
            out = self.vars[name]
        except KeyError:
            out = None
        return out

    def set(self, name: str, value: Any) -> None:
        self.vars[name] = value

    def print(self, *args):
        logging.info(*args)

    def welcome(self):
        self.print("DeepFold v2")

    def run(self) -> None:
        self.welcome()

        for line in self.commands:
            cmd = line["command"]
            if cmd == "variable":
                self.set(line["id"], line["value"])
            elif cmd == "model":
                if isinstance(line["models"], str):
                    models = [line["models"]]
                elif isinstance(line["models"], list):
                    models = line["models"]
                else:
                    TypeError("Invalid type")
                self.set(line["id"], models)
            elif cmd == "entity":
                pass
            elif cmd == "predict":
                pass
            elif cmd == "graph":
                pass
            else:
                ValueError("Unknown command '{cmd}'")

        print("=== Variables ===")
        for k, v in self.vars.items():
            print(k, "->", v)

    def parse(self, s: str) -> None:
        self.commands = self.parser.parse(s)


def main():
    interpreter = Interpreter(debug=False)
    interpreter.parse(open(sys.argv[1]).read())
    interpreter.run()


@dataclass
class Model:
    id: str
    models: List[str] = field(default=list())


@dataclass
class Entity:
    name: str
    options: Any


@dataclass
class Graph:
    id: str
    graph: nx.Graph = field(default=nx.Graph())


@dataclass
class Predict:
    name: str
    model: Model
    stoi: List[Tuple[Entity, int]]
    options: Dict[str, Any]
