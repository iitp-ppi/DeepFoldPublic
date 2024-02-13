import glob
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

SchemeType = Dict[str, List[str]]


class SecondPair:
    def __init__(self, first: str, second: str):
        assert type(first) == str
        assert type(second) == str

        self.first = first
        self.second = second

    def __hash__(self) -> int:
        hash(self.second)


class SchemeRegularizer:
    """A class for regularizing a scheme based on given key order and base directory."""

    def __init__(
        self,
        key_order: Iterable[str],  # Iterable containing keys in the desired order
        base_dir: Optional[Union[str, Path]] = None,  # Base directory to search for paths
    ) -> None:
        """
        Initializes the SchemeRegularizer.

        Parameters:
        - key_order: An iterable containing strings representing keys in the desired order.
        - base_dir: Optional. A string or Path representing the base directory to search for paths.
                    Defaults to current working directory if not provided.
        """
        self.key_order = list(key_order)  # Convert key_order to a list for faster access
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()  # Convert base_dir to Path object

    def process(self, scheme: SchemeType) -> SchemeType:
        """
        Regularizes the given scheme according to the specified key order.

        Parameters:
        - scheme: A dictionary representing the scheme to be regularized.

        Returns:
        - A new dictionary representing the regularized scheme.
        """
        # Check if keys in the scheme match the specified key order
        assert set(scheme.keys()) == set(self.key_order)

        pairs = set()
        for key in self.key_order:
            for regex in scheme[key]:
                for path in glob.glob(regex, root_dir=self.base_dir, recursive=False):
                    # Add a pair consisting of the current key and the found path to the set
                    pairs.add(SecondPair(key, path))

        new_scheme = defaultdict(list)
        for pair in pairs:
            new_scheme[pair.first].append(pair.second)

        return dict(new_scheme)
