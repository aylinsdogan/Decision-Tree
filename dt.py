from typing import List

class Best():
    def __init__(self,feature_index=None, threshold =None, leftFeatures =None, rightFeatures=None, leftLabels =None, rightLabels=None, gain=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.leftFeatures = leftFeatures
        self.rightFeatures = rightFeatures
        self.leftLabels = leftLabels
        self.rightLabels = rightLabels
        self.gain = gain
class Node():
    def __init__(self,feature_index=None, left =None, right =None,threshold=None, gain=None, leaf=None):
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.threshold = threshold
        self.gain = gain
        self.leaf = leaf

class DecisionTreeClassifier:
    def __init__(self, max_depth: 5):
        self.root = None
        self.max_depth = max_depth

    def fit(self, X: List[List[float]], y: List[int]):
        self.root = self.fitAux(X, y, 0) 

    def fitAux(self, X: List[List[float]], y: List[int], depth):

        if depth < self.max_depth:
            best = self.bestSplit(X,y)
            if(best.gain > 0): 

                left= self.fitAux(best.leftFeatures,best.leftLabels, depth + 1)
                right = self.fitAux(best.rightFeatures,best.rightLabels, depth + 1)

                return Node(best.feature_index,left, right, best.threshold, best.gain)

        temp = [0]*3
        mx=-1
        for i in y:
            temp[i] += 1
        value = 0
        for i in temp:
            if(i > mx):
                mx=i
        for i in range(len(temp)):
            if(temp[i] == mx):
                break
            value = value+1

        return Node(leaf=value)
    def bestSplit(self,X,Y):
        if(len(X) == 0):
            return Best()
        featureNum = len(X[0])
        maxGain = -100000000
        best = Best(gain=-1)
        values = []
        for fi in range(featureNum):
            for j in X: 
                values.append(j[fi])
            thresholds = self.unique(values)
            for th in thresholds: 
                x_left, x_right, y_left, y_right = self.split(X,Y,fi,th)

                if len(x_left) > 0 and len(x_right) > 0:
                    gain = self.information_gain(Y, y_left, y_right)
                    if gain > maxGain: 
                        best.feature_index = fi
                        best.threshold = th
                        best.gain = gain
                        best.leftFeatures = x_left
                        best.rightFeatures = x_right
                        best.leftLabels = y_left
                        best.rightLabels = y_right
                        maxGain = gain
        return best

    def split(self, X, Y, feature_index, threshold):
        x_right = []
        x_left = []
        y_right = []
        y_left = []
        c = 0
        for i in X:
            if (i[feature_index] < threshold):
                x_left.append(i)
                y_left.append(Y[c])
            else:
                x_right.append(i)
                y_right.append(Y[c])
            c = c + 1
        return x_left, x_right, y_left, y_right

    def unique(self, y):
        uniq = []
        for i in y:
            boo = True
            for j in uniq:
                if (i == j):
                    boo = False
            if (boo):
                uniq.append(i)
        return uniq

    def giniComp(self, y):
        gini = 0
        label = self.unique(y) 
        temp = [0] * 3
        for i in y:
            temp[i] += 1
        for i in label:
            p = temp[i] / len(y)
            gini += p ** 2
        return 1 - gini

    def information_gain(self, parent, left, right):
        left_weight = len(left) / len(parent)
        right_weight = len(right) / len(parent)
        gain = self.giniComp(parent) - (left_weight * self.giniComp(left) + right_weight * self.giniComp(right))
        return gain

    def predict(self, X: List[List[float]]):
        preditcs = []
        for x in X:
            preditcs.append(self.predictAux(self.root,x))
        return preditcs

    def predictAux(self,node,x):
        if node.leaf != None:
            return node.leaf
        index = node.feature_index
        feature = x[index]
        if feature <= node.threshold:
            return self.predictAux(node.left, x)
        else:
            return self.predictAux(node.right, x)

    def print_tree(self, tree=None, indent=" "):

        if not tree:
            tree = self.root

        if tree.leaf is not None:
            print(tree.leaf)

        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)


# if __name__ == '__main__':
# X, y = ...
# X_train, X_test, y_train, y_test = ...

# clf = DecisionTreeClassifier(max_depth=5)
# clf.fit(X_train, y_train)
# yhat = clf.predict(X_test)
