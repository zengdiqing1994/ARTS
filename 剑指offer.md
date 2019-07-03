[数组](#数组)

[字符串](#字符串)

[链表](#链表)

[树](#树)

[栈和队列](#栈和队列)

### 数组

**3.数组中的重复数字**

思路1：

先排序，再从排好序的数组中找出重复的数字，O(nlogn)

```py
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # 列表排序
        numbers.sort()
        for i in range(len(numbers)):
            # 判断是否扫描到列表末尾。防止溢出。
            if i == len(numbers) - 1:
                return False
            if numbers[i] == numbers[i+1]:
                duplication[0] = numbers[i]
                return True
        return False
```


思路2：

用python中的字典，从头到尾按顺序扫描数组的每个数字，每扫描到一个数字的时候，判断哈希表里是否已经包含了该数字。如果哈希表里还没有这个数字，就把它加入
哈希表。如果哈希表里已经存在该数字，就找到一个重复的数字。

```py
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        dict = {}
        for num in numbers:
            if num not in dict:
                dict[num] = 0
            else:
                duplication[0] = num
                return True
        return False
```

思路3：

重新排列这个数组，从头到尾依次扫描数字，当扫描搭配下标为i的数字时，首先比较这个数字numbers[i]是不是等于i，如果是，就扫描下一个数字；如果不是，就再
拿他和第number[i]个数字进行比较。如果他和第numbers[i]个数字相等，就找到了一个重复的数字(该数字在下标为i和numbers[i]的位置都出现了。)。如果他和
第numbers[i]个数字不等，就把第i个数字和第numebrs[i]个数字交换，把numbers[i]放在属于他的位置。接下来重复这个比较，交换的过程，直到找到一个重复的
数字。

```py
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        if numbers == None or len(numbers)<0:
            return False
        for i in range(len(numbers)):
            while numbers[i] != i:
                if numbers[i] == numbers[numbers[i]]:
                    duplication[0] = numbers[i]
                    return True
                temp = numbers[i]
                numbers[i] = numbers[temp]
                numbers[temp] = temp
                #numbers[i],numbers[numbers[i]]=numbers[numbers[i]],numbers[i]
        return False
```
注意这里不能直接用python的直接赋值代替交换swap，会超时。

**4.二维数组中的查找**

思路：

首先选取数组中右上角的数字，如果该数字等于要查找的数字，则查找过程结束；如果数字大于要查找的数字，则剔除这个数字所在的列；如果该数字小于要查找的数字，则
剔除这个数字所在的行。也就是说，如果要查找的数字不在数组的右上角，则每一次都在数组的查找范围中剔除一行或者一列，这样每一步都可以缩小查找的范围，直到找到
查找的数字。

```py
class Solution:
    # array 二维列表
    def Find(self, target, array):
        if array == []:
            return False
        num_row = len(array)
        num_col = len(array[0])
        
        i = num_col - 1
        j = 0
        while i>=0 and j<num_row:
            if array[j][i] > target:
                i-=1
            elif array[j][i] < target:
                j+=1
            else:
                return True
```

**21.调整数组顺序使得奇数位于偶数前面**

思路：遍历数组，奇数前插入，偶数后面插入。这里使用sorted排序，python简直了。

```py
def reOrderArray(self,array):
    return sorted(array, key = lambda c:c%2, reversed=True)
```

**39.数组中出现次数超过一半的数字**

思路：多数投票问题，可以利用时间复杂度维O（N）方法来解决这个问题。

使用cnt来统计一个元素出现的次数，当遍历到的元素和统计元素相等时，令cnt++，否则令cnt--，如果前面查找了i个元素，且cnt==0，说明前i个元素没有majority，或者
有majority，但是次数少于i/2，因为如果多于i/2的话cnt就一定不会为0.此时剩下的n-i个元素中，majority的数目依然多余(n-i)/2，因此继续查找就能找出majority

```py
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        length = len(numbers)
        if not numbers:
            return 0
        result = numbers[0]
        times = 1
        for i in range(1,length):
            if times == 0:
                result = numbers[i]
                times = 1
            elif numbers[i] == result:
                times += 1
            else:
                times -= 1
        if not self.CheckNoreThanHalf(numbers,length,result):
            return 0
        return result
        
    def CheckNoreThanHalf(self,numbers,length,number):
        times = 0
        for i in range(length):
            if numbers[i] == number:
                times += 1
        if times*2 <= length:
            return False
        return True
```

### 字符串



### 链表

**18.（1）删除链表中的节点**

思路:

我们指定要删除node指向的那个值，我们无法拿到之前的节点，只能操作当前的节点。我们将下一个节点的值赋值给当前节点，这样看上去已经删除了，删除掉一个就好.

```py
class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        if node==None:#注意边界
            return
        if node.next==None:#代表node是最后一个结点
            del node  #删除，将node设定为none
            node = None
            return
        
        node.val = node.next.val
        delNode = node.next
        node.next = delNode.next
        del delNode
        return 
```
**18.(2)删除链表中的重复节点**

思路：需要确定删除函数的参数，这个函数需要输入待删除链表的头节点。头节点可能与后面的节点重复，也就是说头节点也可能被删除。接下来从头遍历整个链表。如果当前节点first与下一个节点pHead相同时，那么就是重复的节点，都可以被删除。为了保证删除之后的链表人然是相连的，我们要把当前节点的前一个节点和后面值比当前节点的值大的节点相连。要确保之前的节点始终与下一个没有重复的节点连在一起。

1——2——3——3——4——4——5

1——2——5

比如，遍历到第一个值为3的时候，前一个节点为2.接下来还是3,这两个节点都应该被删除，所以前一个节点2应该和4相连，由于4也是两个，所以还是会删除，所以最终2会合5相连。

```py
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if pHead is None or pHead.next is None:
            return pHead
        first = ListNode(-1)
        first.next = pHead
        last = first
        while pHead and pHead.next:
            if pHead.val == pHead.next.val:
                val = pHead.val
                while pHead and val == pHead.val:
                    pHead = pHead.next
                last.next = pHead
                
            else:
                last = pHead
                pHead = pHead.next
        return first.next
```




**22.链表中倒数第k个结点**

思路：

设链表的长度为N，设两个指针p1,p2,先让p1移动k个节点，则还有N-K个节点可以移动。此时让p1和p2同时移动，可知当p1移动到链表结尾时，p2移动到N-K个节点处
，该位置就是倒数第k个节点。

```py
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if head == None or k <= 0:
            return None
        
        pAhead = head
        pBhead = None
        
        for i in range(k-1):
            if pAhead.next != None: #先让p1指针移动
                pAhead = pAhead.next
            else:
                return None
        pBhead = head   
        while pAhead.next != None:     #再同时移动p1,p2,最终p1到终点的时候就是倒数第k个
            pAhead = pAhead.next
            pBhead = pBhead.next
        return pBhead
```

**23.链表中环的入口点**

思路：我们可以使用双指针，一个指针fast每次移动两个节点，一个指针slow每次移动一个节点，因为存在环，所以两个指针必定相遇在环中的某个节点上。

```py
class Solution:
    def MeetNode(self,head):
        if not head:
            return None
        slow = head.next
        if slow == None:
            return None
        fast = slow.next
        while fast:
            if slow == fast:
                return slow
            slow = slow.next
            fast = fast.next.next
            
    def EntryNodeOfLoop(self, pHead):
        # write code here
        meetNode = self.MeetNode(pHead)
        if not meetNode:
            return None:
        loop = 1
        flag = meetNode
        while flag.next != meetNode:
            loop += 1
            flag = flag.next
            
        fast = pHead
        for i in range(loop):
            fast = fast.next
        slow = pHead
        while fast != slow:
            fast = fast.next
            slow = slow.next
        return fast
```

**24.反转链表**

思路：链表这一类题目比较需要指针的参与，把各个结点的指向反转一下的话，我们就需要有三个指针，pre，cur，nxt，我们遍历的时候需要把这三个指针依次往后挪，然后再让cur.next指向前继结点pre，依次循环，直到nxt为None。注意，如果不在循环里面，我们也需要做一次反向指针的操作。

```py
class Solution:
    def ReversedList(self,head):
        if head is None:
            return None
        pre = None
        cur = head
        nxt = cur.next
        while nxt:
            cur.next = pre
            pre = cur
            cur = nxt
            nxt = nxt.next
        cur.next = pre
        head = cur
        return cur
```

**25.合并两个有序链表**

思路：如果链表1的头结点的值小于链表2的头结点的值，所以链表1的头结点是合并后链表的头结点。继续合并，依然是排好序的，所以合并的步骤和之前是一样的，当我们得到两个链表中值较小的头结点并把它连接到已经合并的链表之后，两个链表还是有序的，这是典型的递归过程。

```py
def Merge(self,pHead1,pHead2):
    if pHead1 == None:
        return pHead2
    if pHead2 == None:
        return pHead1
    pMergeHead = None
    if pHead1.val < pHead2.val:
        pMergeHead = pHead1
        pMergeHead.next = self.Merge(pHead1.next,pHead2)
    else:
        pMergeHead = pHead2
        pMergeHead.next = self.Merge(pHead1,pHead2.next)
     return pMergeHead
```
**52.两个链表的第一个公共节点**

思路：设置A的长度是a+c，B的长度是b+c，其中c是尾部公共部分的长度。

当访问链表A的指针访问到链表尾部时，令它从链表A的头部重新开始访问链表B；同样的，当访问链表B的指针访问到链表尾部时，令它从链表A的头部重新开始访问A，这样就能控制访问A和B的两个链表指针能同时访问到交点。

```py
def FindFirstCommonNode(self,pHead1,pHead2):
    if not pHead1 or not pHead2:
        return None
    p1,p2 = pHead1,pHead2
    len1 = len2 = 0
    while p1:
        len1 += 1
        p1 = p1.next
    whiel p2:
        len2 += 1
        p2 = p2.next
    if len1 > len2:
        while len1 - len2:
            pHead1 = pHead1.next
            len1 -= 1
    else:
        while len2-len1:
            pHead2 = pHead2.next
            len2 -= 1
    while pHead1 and pHead2:
        if pHead1 is pHead2:
            retun pHead1
        pHead1 = pHead1.next
        pHead2 = pHead2.next
    return None

```

### 树

**7.重建二叉树**

思路：我们知道前序遍历的第一个值为根节点的值，使用这个值将中序遍历结果分成两个部分，左部分为树的左子数中序遍历结果，右部分为树的右子树中序遍历的结果.
就可以用递归的手段解决


```py
class TreeNode:
    def __init__(self, x):      # 初始化树
        self.val = x
        self.left = None
        self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if not pre and not tin:
            return None
        root = TreeNode(pre[0])     # 前序遍历的root
        if set(pre) != set(tin):    #判断其他情况
            return None
        i = tin.index(pre[0])          #看前序遍历的root在中序遍历的哪个位置
        root.left = self.reConstructBinaryTree(pre[1:i+1],tin[:i]) #开始递归
        root.right = self.reConstructBinaryTree(pre[i+1:],tin[i+1:])
        return root
```



### 栈和队列
