class TreeNode:
    def __init__(self, group=None, column=None):
        self.group = group
        self.column = column


class Group:
    def __init__(self, name=None, value=None, status=None):
        self.name = name
        self.value = value
        self.status = status
