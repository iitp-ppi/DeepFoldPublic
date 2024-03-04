import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import networkx as nx

from deepfold.launch.parser import Parser

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
        self.name_map = dict()

        # Parser
        self.parser = Parser(debug=self.debug)

    def get(self, name: str) -> Any:
        if name.startswith("$"):
            name = name[1:]
        return self.name_map[name]

    def set(self, name: str, value: Any) -> None:
        if name.startswith("$"):
            name = name[1:]
        self.name_map[name] = value

    @dataclass
    class ID:
        name: str
        ref: Any

    def run(self) -> None:
        pass
