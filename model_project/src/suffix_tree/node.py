class TreeNode: 

    def __init__(self, value=None, prob=dict, suffix=None):
        self.value = value
        self.prob = prob #dict where the key are the next symbols and the values their probabilities
        self.suffix = suffix # fatherNode Value
        self.children = []

    def showInfo(self):
            print(f"Node value: {self.value}, Prob: {self.prob}, Suffix: {self.suffix}, Children: {list(self.children.keys())}") 

    def addChild(self, childNode):
         self.children.append(childNode)
    
    def setValue(self, value):
        self.value = value
    
    def setProb(self, prob):   
        self.prob = prob
    
    def removeChild(self, childNode):
        self.children.remove(childNode)

    def is_leaf(self):
        return len(self.children) == 0
    
