from sklearn import datasets

iris = datasets.load_iris()


class Node:
    def __init__(self):
        self.gini: float
        self.value: list
        self.class_: str
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None



for d in iris.data:
    print(d)


root = Node()


def construct_tree(depth=2):
