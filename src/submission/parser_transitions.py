class PartialParse(object):
    """
    Represents a partial parse during a transition-based dependency parsing process.
    Maintains the stack, buffer, and dependencies to track parsing progress.
    """

    def __init__(self, sentence):
        """
        Initializes the state of the partial parse for a given sentence.

        Attributes:
            stack (list): The parsing stack initialized with a single "ROOT".
            buffer (list): The buffer initialized with the words of the input sentence.
            dependencies (list): List of dependencies produced during parsing. Each dependency
                                 is a tuple of the form (head, dependent).

        Args:
            sentence (list): The input sentence to be parsed. This should remain unmodified.
        """
        self.sentence = sentence  # Store the sentence for reference
        self.stack = ["ROOT"]  # Initialize stack with ROOT
        self.buffer = list(sentence)  # Initialize buffer with the input sentence
        self.dependencies = []  # Initialize an empty list of dependencies

    def parse_step(self, transition):
        """
        Applies a single transition to the current partial parse state.

        Args:
            transition (str): A transition command ("S", "LA", "RA") representing
                              Shift, Left-Arc, and Right-Arc operations.
        """
        if transition == "S":  # Shift operation
            if self.buffer:  # Ensure buffer is not empty
                self.stack.append(self.buffer.pop(0))
        elif transition == "LA":  # Left-Arc operation
            if len(self.stack) > 1:  # Ensure enough items in the stack
                head = self.stack[-1]
                dependent = self.stack[-2]
                self.dependencies.append((head, dependent))
                self.stack.pop(-2)
        elif transition == "RA":  # Right-Arc operation
            if len(self.stack) > 1:  # Ensure enough items in the stack
                head = self.stack[-2]
                dependent = self.stack[-1]
                self.dependencies.append((head, dependent))
                self.stack.pop()

    def parse(self, transitions):
        """
        Executes a sequence of transitions to parse the input sentence.

        Args:
            transitions (list): A sequence of transitions to be applied.

        Returns:
            dependencies (list): List of dependencies produced by parsing the sentence.
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, device, batch_size):
    """
    Parses multiple sentences in minibatches using the given parsing model.

    Args:
        sentences (list): A list of sentences to parse. Each sentence is a list of words.
        model: A model with a `predict` method to determine the next transition for each parse.
        device (str): The device to use for computation.
        batch_size (int): The number of PartialParse objects to process in each batch.

    Returns:
        dependencies (list): A list where each element contains the dependencies
                             for a parsed sentence in the order of the input sentences.
    """

    # Initialize PartialParse objects for each sentence
    partial_parses = [PartialParse(sentence) for sentence in sentences]

    # Keep track of unfinished parses
    unfinished_parses = partial_parses[:]

    while unfinished_parses:
        # Take a minibatch of size batch_size (or less if at the end)
        minibatch = unfinished_parses[:batch_size]

        # Get the model's predicted transitions for the minibatch
        transitions = model.predict(minibatch, device)

        # Apply the predicted transitions to each corresponding PartialParse object
        for i in range(len(minibatch)):
            minibatch[i].parse_step(transitions[i])

        # Remove completed parses (buffer is empty and only ROOT remains in the stack)
        unfinished_parses = [
            pp for pp in unfinished_parses if pp.buffer or len(pp.stack) > 1
        ]

    # Collect dependencies for each sentence
    dependencies = [pp.dependencies for pp in partial_parses]

    return dependencies


def test_step(name, transition, stack, buf, deps, ex_stack, ex_buf, ex_deps):
    """Utility to verify parse_step behavior matches expected results."""
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps
    pp.parse_step(transition)
    assert (
        tuple(pp.stack) == ex_stack
    ), f"{name} test: stack {pp.stack}, expected {ex_stack}"
    assert (
        tuple(pp.buffer) == ex_buf
    ), f"{name} test: buffer {pp.buffer}, expected {ex_buf}"
    assert (
        tuple(sorted(pp.dependencies)) == ex_deps
    ), f"{name} test: deps {pp.dependencies}, expected {ex_deps}"
    print(f"{name} test passed!")


def test_parse_step():
    """Tests the parse_step method of PartialParse."""
    test_step(
        "SHIFT",
        "S",
        ["ROOT", "the"],
        ["cat", "sat"],
        [],
        ("ROOT", "the", "cat"),
        ("sat",),
        (),
    )
    test_step(
        "LEFT-ARC",
        "LA",
        ["ROOT", "the", "cat"],
        ["sat"],
        [],
        (
            "ROOT",
            "cat",
        ),
        ("sat",),
        (("cat", "the"),),
    )
    test_step(
        "RIGHT-ARC",
        "RA",
        ["ROOT", "run", "fast"],
        [],
        [],
        (
            "ROOT",
            "run",
        ),
        (),
        (("run", "fast"),),
    )


def test_parse():
    """Tests the parse method of PartialParse."""
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (("ROOT", "parse"), ("parse", "sentence"), ("sentence", "this"))
    assert (
        dependencies == expected
    ), "parse test resulted in dependencies {:}, expected {:}".format(
        dependencies, expected
    )
    assert tuple(sentence) == (
        "parse",
        "this",
        "sentence",
    ), "parse test failed: the input sentence should not be modified"
    print("parse test passed!")


class DummyModel(object):
    """A simple mock model to test the minibatch_parse function."""

    def predict(self, partial_parses, device):
        return [
            ("RA" if pp.stack[1] == "right" else "LA") if len(pp.buffer) == 0 else "S"
            for pp in partial_parses
        ]


def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert (
        deps == ex_deps
    ), "{:} test resulted in dependency list {:}, expected {:}".format(
        name, deps, ex_deps
    )


def test_minibatch_parse():
    """Tests the minibatch_parse function."""
    device = "cpu"
    sentences = [
        ["right", "arcs", "only"],
        ["right", "arcs", "only", "again"],
        ["left", "arcs", "only"],
        ["left", "arcs", "only", "again"],
    ]
    deps = minibatch_parse(sentences, DummyModel(), device, 2)
    test_dependencies(
        "minibatch_parse",
        deps[0],
        (("ROOT", "right"), ("arcs", "only"), ("right", "arcs")),
    )
    test_dependencies(
        "minibatch_parse",
        deps[1],
        (("ROOT", "right"), ("arcs", "only"), ("only", "again"), ("right", "arcs")),
    )
    test_dependencies(
        "minibatch_parse",
        deps[2],
        (("only", "ROOT"), ("only", "arcs"), ("only", "left")),
    )
    test_dependencies(
        "minibatch_parse",
        deps[3],
        (("again", "ROOT"), ("again", "arcs"), ("again", "left"), ("again", "only")),
    )
    print("minibatch_parse test passed!")


if __name__ == "__main__":
    test_parse_step()
    test_parse()
    test_minibatch_parse()
