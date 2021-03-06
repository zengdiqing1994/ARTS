﻿# Algorithm
[七大排序算法](#七大排序算法)

[数组字符串](#数组字符串)

[链表](#链表)

[堆/栈/队列](#堆/栈/队列)

[二叉树](#二叉树)

[动态规划](#动态规划)



### 七大排序算法

#### 1.快速排序

```py
class QuickSort:
    def quickSort(self, A, n):
        if n<1:
            return A
        left = []
        right = []
        pivot = A.pop()
        for i in A:
            if i<pivot:
                left.append(i)
            else:
                right.append(i)
        result = self.quickSort(left,len(left)) + [pivot] + self.quickSort(right,len(right))
        return result
```

#### 2.归并排序

```py
Class MergeSort:
    def mergeSort(self,A,n):
        if n <= 1:
            return A
        mid = len(A)//2
        left = mergeSort(A[:mid],mid)
        right = mergeSort(A[mid:],mid)
        i,j = 0,0
        result = []
        while i<len(left) and j<len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i+=1
            else:
                result.append(right[j])
                j+=1
        result += left[i:]
        result += right[j:]
        return result
```

#### 3.冒泡排序

```py
class BubbleSort:
    def bubbleSort(self,A,n):
        for i in range(len(A)):
            for j in range(len(A)-i):
                if A[j]>A[j+1]:
                    A[j],A[j+1] = A[j+1],A[j]
        return A
```

#### 4.选择排序

```py
class SelectionSort:
    def selectionSort(self,A,n):
        for i in range(len(A)):
            minIndex = i
            for j in range(i+1,len(A)):
                if A[j]<A[minIndex]:
                   minIndex = j
            if i != minIndex:
                A[i],A[minIndex] = A[minIndex],A[i]
```

#### 5.插入排序

```py
Class InsertSort:
    def insertSort(self,A,n):
        for i in range(n):
            for j in range(i,0,-1):
                if A[j]<A[j-1]:
                    A[j],A[j-1]=A[j-1],A[j]
                else:
                    break
        return A
```

#### 6.希尔排序

```py
class ShellSort:
    def shellSort(self, A, n):
        gap = n//2
        while gap >= 1:
            for i in range(gap,n):
                while (i-gap)>=0:
                    if A[i] < A[i-gap]:
                        A[i],A[i-gap] = A[i-gap],A[i]
                        i -= gap
                    else:
                        break
            gap //= 2
        return A
```

#### 7.堆排序

```py
class HeapSort:
    def heapSort(self, A, n):
        # write code here
        for i in range(n/2+1, -1, -1):
            self.MaxHeapFixDown(A, i, n);
        for i in range(n-1, -1, -1):
            A[0], A[i] = A[i], A[0]
            self.MaxHeapFixDown(A, 0, i)
        return A
      
    def MaxHeapFixDown(self, A, i, n):
        tmp = A[i]
        j = 2*i+1
        while(j<n):
            if j+1<n and A[j+1] > A[j]:
                j+=1
            if A[j] < tmp:
                break
            A[i] = A[j]
            i = j
            j = 2*i+1
        A[i] = tmp
```

**347.前K个高频元素**

给定一个非空的整数数组，返回其中出现频率前 k 高的元素。

示例 1:

输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
示例 2:

输入: nums = [1], k = 1
输出: [1]

思路：可用优先队列，以及桶排序

```py
def topFrequency(self,nums,k):
    count_list = {}
    result = []
    if i in nums:
        count_list[i] = count_list.get[i,0] + 1
        t = sorted(count_list.items(),key = lambda l:l[1],reverse = True)
        for i in range(k):
            result.append(t[i][0])
        return result
```


**215.在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。**

示例 1:

输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
示例 2:

输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
说明:

你可以假设 k 总是有效的，且 1 ≤ k ≤ 数组的长度。

思路：

用到QuickSort，排序的方向是从大到小，每次都找一个枢纽pivot，然后遍历其他所有的数字，像这道题从大到小排，把大于中枢点的数放在左边，小于中枢点的放在右边，这样中枢点就是是整个数组中第几大的数字就确定了，虽然左右两边不一定完全有序，但是不影响。如果求出pivot正好是k-1,就求到了，如果pivot大于k-1,就说明在要求的数字在左半边部分，更新右边界，反之更新左边界。

```py
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        left, right = 0,len(nums)-1
        while nums:
            pos = self.partition(nums,left,right)
            if pos == k-1:
                return nums[pos]
            elif pos > k-1:
                right = pos-1
            else:
                left = pos+1      
        
    def partition(self,nums,left,right):    
        pivot = nums[left]
        l = left + 1
        r = right
        while(l<=r):
            if nums[l]<pivot and nums[r]>pivot:
                # l += 1
                # r -= 1
                nums[l],nums[r] = nums[r],nums[l]
            if nums[l] >= pivot:
                l+=1
            if nums[r] <= pivot:
                r-=1
        nums[left],nums[r]=nums[r],nums[left]
        return r                
```

### 数组字符串

**1.两数之和**

给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

示例:

给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]

思路：hash表，也就是dict字典，target=9,nums[]=2,tmp=9-2=7,如果2不在dict里面，就把i=0传给dic[7]，就是{7:0}，index是0。如果num[1]也就是7在dic里面，
就返回dic[7]对应的0的index，i就是1

```py
def twoSum(nums,target):
    dic = {}
    for i in range(len(nums)):
        tmp = target - nums[i]
        if nums[i] in dic:
            return dic[nums[i]],i
        else:
            dic[tmp] = i
```

**15. 三数之和**

给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]

思路：排序，固定左边，如果左边重复，继续，左右弄边界，去重，针对不同的左右边界情况处理

即a+b+c>0 动c往左，a+b+c<0 动b往右

```py
class Solution:
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []	#结果的数组
        nums.sort()	#先排序
        for i in range(len(nums)-2):
            if i>0 and nums[i] == nums[i-1]:	#如果有重复的，去重
                continue
            l = i+1	#两个指针
            r = len(nums) - 1
            while l<r:
                s = nums[i] + nums[l] + nums[r]	#求的和是三个数相加
                if s < 0:	#若和小于0，就让l右移
                    l+=1
                elif s > 0:	#和大于0，就让r左移
                    r-=1
                else:
                    res.append((nums[i],nums[l],nums[r]))	#若等于零，就把三个数放入数组中
                    while l<r and nums[l] == nums[l+1]:#左边两个数相同，就跳过
                        l+=1
                    while l<r and nums[r] == nums[r-1]:#右边同理
                        r-=1
                    l+=1;r-=1
        return res
```

**6.将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。**

比如输入字符串为 "LEETCODEISHIRING" 行数为 3 时，排列如下：

L   C   I   R
E T O E S I I G
E   D   H   N
之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："LCIRETOESIIGEDHN"。

请你实现这个将字符串进行指定行数变换的函数：

string convert(string s, int numRows);
示例 1:

输入: s = "LEETCODEISHIRING", numRows = 3
输出: "LCIRETOESIIGEDHN"
示例 2:

输入: s = "LEETCODEISHIRING", numRows = 4
输出: "LDREOEIIECIHNTSG"
解释:

L     D     R
E   O E   I I
E C   I H   N
T     S     G

思路：idx从0开始，自增直到numRows-1，此后又一直自减到0，重复执行。

从第一行开始往下，走到第四行又往上走，这里用 step = 1 代表往下走， step = -1 代表往上走

因为只会有一次遍历，同时把每一行的元素都存下来，所以时间复杂度和空间复杂度都是 O(N)

```py
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows==1 or numRows>=len(s):
            return s                #判断情况
        res = [''] * numRows        #初始化res结果
        idx, step = 0, 1            
        for c in s:             
            res[idx] += c           #把字符加入res中
            if idx == 0:                
                step = 1            #step向下加一
            elif idx == numRows-1:  #一直到最后一行为止
                step = -1           #向上操作
            idx += step             #idx代表第几行
        return ''.join(res)         
```

**53.最大子序和**

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example:

Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
Follow up:

If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

思路：动态规划（只关注：当然值 和 当前值+过去的状态，是变好还是变坏，一定是回看容易理解）

ms(i) = max(ms[i-1]+ a[i],a[i])

到i处的最大值两个可能，一个是加上a[i], 另一个从a[i]起头，重新开始。可以AC

```py
def maxSubArray(nums):
    n = len(nums)
    maxSum = [nums[0] for i in range(n)]
    for i in range(1,n):
        maxSum[i] = max(maxSum[i-1] + nums[i],nums[i])
    return max(maxSum)
```


**33.假设按照升序排序的数组在预先未知的某个点上进行了旋转。**

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

你可以假设数组中不存在重复的元素。

你的算法时间复杂度必须是 O(log n) 级别。

示例 1:

输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4
示例 2:

输入: nums = [4,5,6,7,0,1,2], target = 3
输出: -1

思路：二分法

这道题让在旋转数组中搜索一个给定值，若存在返回坐标，若不存在返回-1。我们还是考虑二分搜索法，但是这道题的难点在于我们不知道原数组在哪旋转了，我们还是用题目中给的例子来分析，对于数组[0 1 2 4 5 6 7] 共有下列七种旋转方法：

0　　1　　2　　 **4　　5　　6　　7**

7　　0　　1　　 2　　4　　5　　6

6　　7　　0　　 1　　2　　4　　5

5　　6　　7　　 0　　1　　2　　4

4　　5　　6　　7　　0　　1　　2

2　　4　　5　　6　　7　　0　　1

1　　2　　4　　5　　6　　7　　0

二分搜索法的关键在于获得了中间数后，判断下面要搜索左半段还是右半段，我们观察上面粗体的数字都是升序的，由此我们可以观察出规律，如果中间的数小于最右边的数，则右半段是有序的，若中间数大于最右边数，则左半段是有序的，我们只要在有序的半段里用首尾两个数组来判断目标值是否在这一区域内，这样就可以确定保留哪半边了

```py
def search(nums,target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + ((r-l)>>2)
        if nums[mid] == target:
            return mid
        if nums[mid] < nums[r]:
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
        else:
            if nums[l] <= target < nums[mid]
                r = mid - 1
            else:
                l = mid + 1
    return -1
```



**48.你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。**

示例 1:

给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
示例 2:

给定 matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

原地旋转输入矩阵，使其变为:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]

思路：先用一个临时变量放置非对角线的数字，然后再把几行数组反过来排列

```
def rotate(matrix):
    length = len(matrix)
    for i in range(length):
        for j in range(i+1,length):
            temp = matrix[i][j]
            matrix[i][j] = matrix[j][i]
            matrix[j][i] = temp
    for i in range(length):
        matrix[i] = matrix[i][::-1]
```

**54.给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。**

示例 1:

输入:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
输出: [1,2,3,6,9,8,7,4,5]
示例 2:

输入:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
输出: [1,2,3,4,8,12,11,10,9,5,6,7]

思路：用四个变量来控制辩解，方向总是“左右上下”，这个和Z字形变换很像。

```py
def spiralOrder(matrix):
    if matrix == []:
        return []
    res = []
    maxUp = maxLeft = 0
    maxDown = len(matrix) - 1
    maxRight = len(matrix[0]) - 1
    direction = 0     # 0 go right , 1 go down, 2 go left, 3 up
    while True:
        if direction == 0: # go right
            for i in range(maxLeft,maxRight+1):
                res.append(matrix[maxUp][i])
            maxUp += 1
        elif direction == 1: # 1 go down
            for i in range(maxUp,maxDown+1):
                res.append(matrix[i][maxRight])
            maxRight -= 1
        elif direction == 2: # go left
            for i in reversed(range(maxLeft,maxRight+1)):
                res.append(matrix[maxDown][i])
            maxDown -= 1
        else:           # go up
            for i in reversed(range(maxUp,maxDown+1)):
                res.append(matrix[i][maxLeft])
            maxLeft += 1
        if maxUp > maxDown or maxLeft > maxRight:
            return res
        direction = (direction + 1) % 4         # direction = 3之后就是0重新开始
```
时间复杂度:O(m*n)

空间复杂度：O（1）

**59.给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。**

示例:

输入: 3
输出:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]

思路：和之前的那道题类似，只不过这次要自己生成一个矩阵

```py
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        curNum = 0
        matrix = [[0 for i in range(n)] for j in range(n)]
        maxUp = maxLeft = 0
        maxDown = maxRight = n - 1
        direction = 0
        while True:
            if direction == 0:
                for i in range(maxLeft,maxRight+1):
                    curNum += 1
                    matrix[maxUp][i] = curNum
                maxUp += 1
            elif direction == 1:
                for i in range(maxUp,maxDown+1):
                    curNum += 1
                    matrix[i][maxRight] = curNum
                maxRight -= 1
            elif direction == 2:
                for i in reversed(range(maxLeft,maxRight+1)):
                    curNum += 1
                    matrix[maxDown][i] = curNum
                maxDown -= 1
            else:
                for i in reversed(range(maxUp,maxDown+1)):
                    curNum += 1
                    matrix[i][maxLeft] = curNum
                maxLeft += 1
            if curNum >= n*n:
                return matrix
            direction = (direction + 1) % 4
```

**53.最大子序列的和**

给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

示例:

输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

思路：用DP来求解，只关注：当前值和当前值+过去的状态，是变好还是变坏

状态定义方程：maxSum = [nums[0] for i in range(n)]

状态转移：maxSum[i] = max(maxSum[i-1] + nums[i],nums[i])，一个是加上nums[i]的，另一个是从a[i]起头，重新开始。

```py
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        maxSum = [nums[0] for i in range(n)]
        for i in range(1,n):
            maxSum[i] = max(maxSum[i-1] + nums[i],nums[i])
        return max(maxSum)
```

**76.给定一个字符串 S 和一个字符串 T，请在 S 中找出包含 T 所有字母的最小子串。**

示例：

输入: S = "ADOBECODEBANC", T = "ABC"
输出: "BANC"
说明：

如果 S 中不存这样的子串，则返回空字符串 ""。
如果 S 中存在这样的子串，我们保证它是唯一的答案。

思路：总的来说，我们希望圈定一个最小的窗口，这个窗口里包含所有t中的字符，并且这个窗口的长度要最短。

所以，我们需要边界指针left,right来去圈定我们窗口的范围。

1.先遍历t中字符串，找到各字符出现个次数，储存在hash中。

2.再遍历s中字符串，每遇到一个t中的字符，则把对应hash value - 1，如果这个字符对应的值大于等于0，则count++。这一段我们的目的是划定一个s中的区间，这
个区间包含所有t中字符。count 表示t中有几个字符在s中（当前窗口区间），不包括s中多的重复的字符。

3.当count 第一次等于 t.size()时，说明我们第一次圈定了一个区间，满足所有t中字符在这个区间中都可以找到，但不能保证最短。于是我们更新最短长度，以及最
短字符串。接下来我们要右移我们的窗口了，如果我们这个窗口的第一项（也就是要挪动，要移除的那一项）是组成t所需要的，那我们如果要移除掉它，则hash值要加
一，因为我们当前窗口接下来不会包含那个字符了，同时count也要根据情况减少，因为count表示s窗口中能找到t的几个字符，现在窗口右移，不包含那个必须组件
了，于是要-1。

```py
def solution(s, t):
    res = ""
    if len(s) < len(t):
        return res

    left, right = 0, 0
    min_len = len(s) + 1
    m = {}
    count = 0
    for i in t:
        m[i] = m.get(i, 0) + 1
    while right < len(s):
        if s[right] in m:
            m[s[right]] -= 1
            if m[s[right]] >= 0:
                count += 1
            while (count == len(t)):
                if(right - left + 1 < min_len):
                    min_len = right - left + 1
                    res = s[left:right+1]
                if s[left] in m:
                    m[s[left]] += 1
                    if m[s[left]] > 0:
                        count -= 1
                left += 1
        right += 1
    return res
```

**22. 括号生成**

给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。

例如，给出 n = 3，生成结果为：

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]

思路：非常牛逼的讲解，需要这样的人来给我们讲算法

以Generate Parentheses为例，backtrack的题到底该怎么去思考？
所谓Backtracking都是这样的思路：在当前局面下，你有若干种选择。那么尝试每一种选择。如果已经发现某种选择肯定不行（因为违反了某些限定条件），就返回；如果某种选择试到最后发现是正确解，就将其加入解集

所以你思考递归题时，只要明确三点就行：选择 (Options)，限制 (Restraints)，结束条件 (Termination)。即“ORT原则”（这个是我自己编的）

对于这道题，在任何时刻，你都有两种选择：

加左括号。
加右括号。
同时有以下限制：

如果左括号已经用完了，则不能再加左括号了。
如果已经出现的右括号和左括号一样多，则不能再加右括号了。因为那样的话新加入的右括号一定无法匹配。
结束条件是：
左右括号都已经用完。

结束后的正确性：
左右括号用完以后，一定是正确解。因为1. 左右括号一样多，2. 每个右括号都一定有与之配对的左括号。因此一旦结束就可以加入解集（有时也可能出现结束以后不一定是正确解的情况，这时要多一步判断）。

递归函数传入参数：
限制和结束条件中有“用完”和“一样多”字样，因此你需要知道左右括号的数目。
当然你还需要知道当前局面sublist和解集res。

因此，把上面的思路拼起来就是代码：

if (左右括号都已用完) {
  加入解集，返回
}
//否则开始试各种选择
if (还有左括号可以用) {
  加一个左括号，继续递归
}
if (右括号小于左括号) {
  加一个右括号，继续递归
}
你帖的那段代码逻辑中加了一条限制：“3. 是否还有右括号剩余。如有才加右括号”。这是合理的。不过对于这道题，如果满足限制1、2时，3一定自动满足，所以可以不判断3。

这题其实是最好的backtracking初学练习之一，因为ORT三者都非常简单明显。你不妨按上述思路再梳理一遍，还有问题的话再说。

以上文字来自 1point3arces的牛人解答

```py
class Solution:
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        self.res = []
        self.singleStr('', 0, 0, n)
        return self.res

    def singleStr(self, s, left, right, n):
        if left == n and right == n:
            self.res.append(s)
        if left < n:
            self.singleStr(s + '(',left + 1, right, n)
        if right < left:
            self.singleStr(s + ')',left, right + 1, n)
```


**409.给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。**

在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。

注意:
假设字符串的长度不会超过 1010。

示例 1:

输入:
"abccccdd"

输出:
7

解释:
我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。

思路：这道回文字符串包括以前的回文字符串的题目都比较重要，由于这里的字符串可以打乱，所以问题就转化成了求偶数个字符的个数，我们了解的回文字符串都知道，回文串主要有两种形式，一个是左右完全对称的，比如noon，还有一种是以中心字符为中心，左右对称，比如bob，那么统计出来所有偶数个字符的出现总和，然后如果有奇数个字符的化，我们取出其最大偶数，然后最后结果加上1就可以啦。

```py
class Solution:
    def longestPalindrome(self, s: str) -> int:
        dict1 = {} #用来存储出现过的字符和出现的次数
        j = 0   #存储回文长度
        z = 0   #统计单数次字符的个数
        for i in range(len(s)):
            if s[i] in dict1:
                dict1[s[i]] += 1 #出现的字符作为键，次数作为value值
            else:
                dict1[s[i]] = 1
        for v in dict1: #对构建好的字典进行遍历
            if (dict1[v] + 1) % 2==0:  #出现单数次字符次数减一
                j+=(dict1[v]-1)
                z+=1
            if dict1[v] % 2 == 0: #出现偶数次字符，一定可以构造
                j+=dict1[v]
        if z > 0: 
            return j+1
        else:
            return j
```
这个时间复杂度较高:O(n^2)
空间复杂度O(n)



**2.实现strStr()**

实现 strStr() 函数。

给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。

示例 1:

输入: haystack = "hello", needle = "ll"
输出: 2
示例 2:

输入: haystack = "aaaaa", needle = "bba"
输出: -1
说明:

当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符。

思路：haystack中的字符串和needle的要重合,KMP可以搞定
```py
class Solution:
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        len1 = len(needle)
        if len1 == 0:
            return 0
        for i in range(len(haystack)):
            if haystack[i:i+len1] == needle:
                return i
        return -1
```

**242.有效的异位字符串**

给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的一个字母异位词。

示例 1:

输入: s = "anagram", t = "nagaram"
输出: true
示例 2:

输入: s = "rat", t = "car"
输出: false
说明:
你可以假设字符串只包含小写字母。

思路：使用哈希表，通过统计出现的次数，来判断，数量一致就可以。

```py
class Solution:
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
         dic1, dic2 = {}, {}
         for item in s:
             dic1[item] = dic1.get(item, 0) + 1
         for item in t:
             dic2[item] = dic2.get(item, 0) + 1
            
         return dic1 == dic2
```
时间复杂度：O(s*t)

空间复杂度：O(n)

**3.给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。**

示例 1:

输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

思路：

```py
    def lengthOfLongestSubstring(self, s):
        lookup = collections.defaultdict(int)
        l, r, counter, res = 0, 0, 0, 0 # counter 为当前子串中 unique 字符的数量
        while r < len(s):
            lookup[s[r]] += 1
            if lookup[s[r]] == 1: # 遇到了当前子串中未出现过的字符
                counter += 1
            r += 1
            # counter < r - l 说明有重复字符出现，否则 counter 应该等于 r - l
            while l < r and counter < r - l:
                lookup[s[l]] -= 1
                if lookup[s[l]] == 0: # 当前子串中的一种字符完全消失了
                    counter -= 1
                l += 1
            res = max(res, r - l) # 当前子串满足条件了，更新最大长度
        return res
```

**88. 合并两个有序数组**

给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。

说明:

初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
示例:

输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]

思路：条件是两个有序数组，直接比较nums1和nums2元素的大小，然后根据大小加入到nums1的末尾，最后还要考虑nums2的元素是都还有剩余。

```py
def merge(self, nums1, m, nums2, n):
    pos = m+n-1
    m-=1
    n-=1
    while m >= 0 and n >= 0:
        if nums1[m] > nums2[n]:
            nums1[pos] = nums1[m]
            pos -= 1
            m -= 1
        else:
	    nums1[pos] = nums2[n]
	    pos -= 1
	    n -= 1
    while n >= 0:
        nums1[pos] = nums2[n]
        pos -= 1
        n -= 1
```

**69.实现 int sqrt(int x) 函数。**

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

示例 1:

输入: 4
输出: 2
示例 2:

输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。

```py
  class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        if x == 1:
            return 1
        l, r = 0, x - 1
        while l <= r:
            mid = l + (r-l)//2
            if mid * mid <= x and (mid+1)*(mid+1) > x:
                return mid
            elif mid*mid > x:
                r = mid-1
            else:
                l = mid+1
```


**26.Remove Duplicates from Sorted Array**

Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

Example 1:

Given nums = [1,1,2],

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.

It doesn't matter what you leave beyond the returned length.
Example 2:

Given nums = [0,0,1,1,1,2,2,3,3,4],

Your function should return length = 5, with the first five elements of nums being modified to 0, 1, 2, 3, and 4 respectively.

It doesn't matter what values are set beyond the returned length.

[删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/description/)，因为想用python，第一想到的是
用set函数，set和dict类似，也是一组key的集合，但不存储value。由于key不能重复，所以，在set中，没有重复的key。利用这一特性，可以过滤掉重复元素。

但是仔细一看题目要求，完全不符合，题目是要求原地删除重复元素。如果用set方法，会返回一个新的字典，而原数组不会改变。list有个remove方法，remove方法
是一个没有返回值的原位置改变的方法，他修改了列表但是没有返回值。由于给定的是排序数组，这就容易多了，比较两两相邻的数组元素，相同就把其中的一个利用
remove方法从列表中移除。

```py
class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        while i < len(nums) - 1:
            if nums[i] == nums[i+1]:
                nums.remove(nums[i])
            else:
                i = i+1
        return len(nums)
```

```py
class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        idx = 0
        for num in nums:
            if idx < 1 or num != nums[idx-1]:
                nums[idx] = num
                idx += 1
        return idx
```
特别简单的一题。自己还想了很久。。。


**33. 搜索旋转排序数组**

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

你可以假设数组中不存在重复的元素。

你的算法时间复杂度必须是 O(log n) 级别。

示例 1:

输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4
示例 2:

输入: nums = [4,5,6,7,0,1,2], target = 3
输出: -1

思路：直接用二分法查找数字即可

```py
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = l + ((r-l) >> 2)
            if nums[mid] == target:
                return mid
            if nums[mid] <= nums[r]:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
            else:
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1
```

**75.颜色分类**
给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

注意:
不能使用代码库中的排序函数来解决这道题。

示例:

输入: [2,0,2,1,1,0]
输出: [0,0,1,1,2,2]

```py
class Solution:
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
               
        i, l, r = 0, 0, len(nums) - 1   #定义双指针
        while i < len(nums):        
            if nums[i] == 2 and i < r:  #如果数字是2并且还没到最右边
                nums[i], nums[r] = nums[r], 2   #就把最右边的数字放到i那个位置，然后2放到最右边
                r -= 1              #双指针的右边就往左移
            elif nums[i] == 0 and i > l:    #如果数组数字是0，大于左指针
                nums[i], nums[l] = nums[l], 0   #左指针的数就和到那个位置，0就给左指针位置
                l += 1  #左指针右移一位
            else:
                i += 1
```

**34. 在排序数组中查找元素的第一个和最后一个位置**

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

你的算法时间复杂度必须是 O(log n) 级别。

如果数组中不存在目标值，返回 [-1, -1]。

示例 1:

输入: nums = [5,7,7,8,8,10], target = 8
输出: [3,4]
示例 2:

输入: nums = [5,7,7,8,8,10], target = 6
输出: [-1,-1]

思路：二分法，先找target出现的左边界，判断是否有target后再判断右边界

找左边界：二分，找到一个index，该index对应的值为target，并且它左边index-1对应的值不是target（如果index为0则不需要判断此条件），如果存在index就将其append到
res中，判断此时res是否为空，如果为空，说明压根不存在target，返回[-1, -1]

找右边界：二分，找到一个index（但是此时用于二分循环的l可以保持不变，r重置为len(nums)-1，这样程序可以更快一些），该index对应的值为target

并且它右边index+1对应的值不是target（如果index为len(nums)-1则不需要判断此条件），如果存在index就将其append到res中

```py
class Solution:
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if len(nums)==0:
            return [-1,-1]
        elif target<nums[0] and target<nums[-1]:
            return [-1,-1]
        else:
            l, r = 0, len(nums)-1
        while l<=r:
            mid = (l+r)//2
            if nums[mid]<target:
                l = mid+1
            elif nums[mid]>target:
                r = mid-1
            elif nums[mid] == target:
                l = r = mid
                while l-1>=0 and nums[l-1]==target:
                    l -= 1
                while r+1<=len(nums)-1 and nums[r+1]==target:
                    r += 1
                return [l,r]
        return [-1,-1] 
```

**54. 螺旋矩阵**

给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。

示例 1:

输入:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
输出: [1,2,3,6,9,8,7,4,5]
示例 2:

输入:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
输出: [1,2,3,4,8,12,11,10,9,5,6,7]

思路：四个方向，分别设置不同的direction

```py
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if matrix == []:
            return []
        res = []
        maxUp = maxLeft = 0
        maxDown = len(matrix) - 1
        maxRight = len(matrix[0]) - 1
        direction = 0
        while True:
            if direction == 0:
                for i in range(maxLeft,maxRight+1):
                    res.append(matrix[maxUp][i])
                maxUp += 1
            elif direction == 1:
                for i in range(maxUp,maxDown+1):
                    res.append(matrix[i][maxRight])
                maxRight -= 1
            elif direction == 2:
                for i in reversed(range(maxLeft,maxRight+1)):
                    res.append(matrix[maxDown][i])
                maxDown -= 1
            else:
                for i in reversed(range(maxUp,maxDown+1)):
                    res.append(matrix[i][maxLeft])
                maxLeft += 1
            if maxUp > maxDown or maxLeft > maxRight:
                return res
            direction = (direction + 1) % 4
```


**买卖股票的最佳时机 II**

给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例 1:

输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
示例 2:

输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
示例 3:

输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

思路：这道题比较简单，就是需要如果后面的股价大于前面的就应该买入卖出，反之就不进行操作。
```py
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        benefit = 0
        if len(prices) == 0:
            return 0
        i = 0
        j = 1
        while (j < len(prices)):
            if (prices[j] > prices[i]):
                benefit += prices[j] - prices[i]
            j += 1
            i += 1
        return benefit
```

**第二题：旋转数组**

给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。

示例 1:

输入: [1,2,3,4,5,6,7] 和 k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]

思路：直接用切片，简洁，快速。
```py
class Solution:
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        k = k%len(nums)
        nums[:] = nums[-k:]+nums[:-k]  
```

**1.存在重复**

给定一个整数数组，判断是否存在重复元素。

如果任何值在数组中出现至少两次，函数返回 true。如果数组中每个元素都不相同，则返回 false。

示例 1:

输入: [1,2,3,1]
输出: true
示例 2:

输入: [1,2,3,4]
输出: false
示例 3:

输入: [1,1,1,3,3,4,3,2,4,2]
输出: true

思路：python当中的set集合本身就是没有重复元素的，我们可以利用这个特性去判断真假
```py
class Solution:
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(set(nums)) == len(nums):
            return False
        else:
            return True
```


**2.只出现一次的数字**

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

示例 1:

输入: [2,2,1]
输出: 1
示例 2:

输入: [4,1,2,1,2]
输出: 4

思路：python的特点就是代码少，运算的比较慢，我们可以先把整个数组排序，如果只有一个数，那就是他了，如果不止一个数，看前两个是不是一样，一样的话
就把他们都删除，不一样就取第一个数。
```py
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        for i in range(len(nums)):
            if len(nums) == 1:
                return nums[0]
            if nums[0] == nums[1]:
                del nums[0]
                del nums[0]
            else:
                return nums[0]
```

1.加一
给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储一个数字。

你可以假设除了整数 0 之外，这个整数不会以零开头。

示例 1:

输入: [1,2,3]
输出: [1,2,4]
解释: 输入数组表示数字 123。
示例 2:

输入: [4,3,2,1]
输出: [4,3,2,2]
解释: 输入数组表示数字 4321。

思路：看题目是数组的最后一位加了一个1，所以进位是我们需要考虑的，那么问题来了，倘若是9，进位不就是10了吗？那么就要重新插入一个insert(0,1)
```
class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry = 1
        for i in range(len(digits)-1, -1, -1):
            if digits[i] + carry == 10:
                digits[i] = 0
                carry = 1
            else:
                digits[i] = digits[i] + carry
                carry = 0
        
        if carry == 1:
            digits.insert(0, 1)
        return digits
```

**2.移动零**

给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

示例:

输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
说明:

必须在原数组上操作，不能拷贝额外的数组。
尽量减少操作次数。

思路：pop函数有默认删除末尾的元素，但是只要我们判断列表中出现了0，就记录下下表，就能够把0给pop掉，再append在最后。
```
class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        j=0
        for i in range(len(nums)):
            if nums[j] == 0:
                nums.append(nums.pop(j))

            else:
                j+=1
```

**3.反转字符串**

编写一个函数，其作用是将输入的字符串反转过来。

示例 1:

输入: "hello"
输出: "olleh"
示例 2:

输入: "A man, a plan, a canal: Panama"
输出: "amanaP :lanac a ,nalp a ,nam A"
思路：这道题实在是太简单，用python
```py
class Solution:
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        new_s = ""
        for i in range(len(s)-1,-1,-1):
            new_s += s[i]
        return new_s
```
用在循环里面会让运行时间变得很慢。所以直接御用python的特性，return s[::-1]就好


**4.字符串中的第一个唯一字符**

给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

案例:

s = "leetcode"
返回 0.

s = "loveleetcode",
返回 2.
思路：python中有collections.counter这个函数可以调用，作用是可以统计列表中相同字符的个数。
```py
class Solution:
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        dic = collections.Counter(s)
        for i in range(len(s)):
            if dic[s[i]] == 1:
                return i
        return -1
            
```


 **1.字符串转整数（atoi）**
 
实现 atoi，将字符串转为整数。

在找到第一个非空字符之前，需要移除掉字符串中的空格字符。如果第一个非空字符是正号或负号，选取该符号，并将其与后面尽可能多的连续的数字组合起来，这部分字符即为整数的值。如果第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。

字符串可以在形成整数的字符后面包括多余的字符，这些字符可以被忽略，它们对于函数没有影响。

当字符串中的第一个非空字符序列不是个有效的整数；或字符串为空；或字符串仅包含空白字符时，则不进行转换。

若函数不能执行有效的转换，返回 0。

说明：

假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−231,  231 − 1]。如果数值超过可表示的范围，则返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。

示例 1:

输入: "42"
输出: 42
示例 2:

输入: "   -42"
输出: -42
解释: 第一个非空白字符为 '-', 它是一个负号。
     我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。
示例 3:

输入: "4193 with words"
输出: 4193
解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。
示例 4:

输入: "words and 987"
输出: 0
解释: 第一个非空字符是 'w', 但它不是数字或正、负号。
     因此无法执行有效的转换。
示例 5:

输入: "-91283472332"
输出: -2147483648
解释: 数字 "-91283472332" 超过 32 位有符号整数范围。 
     因此返回 INT_MIN (−231) 。
     
思路：这道题着实是考的编程语言的技巧，我们通过正则表达式可以很好地解决
```py
class Solution:
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        import re#导入正则表达式模块
        #^代表后面的就是开始的，[-+]?说明一开始可以是匹配正负号，也可以没有，\d+代表着后面可以匹配所有的整数
        list_s = re.findall(r"^[-+]?\d+", str.strip()) #strip()是把两边的空格去掉
        if not list_s: 
            return 0
        else:
            num =int(''.join(list_s)) #列表转化为字符串，然后转化为整数
            if num >2**31 -1:
                return 2**31 -1
            elif num < -2**31:
                return -2**31
            else:
                return num
 ```
 
 
 **242.有效字母的异位词**

给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的一个字母异位词。

示例 1:

输入: s = "anagram", t = "nagaram"
输出: true
示例 2:

输入: s = "rat", t = "car"
输出: false

思路：

第一种方法：排序

因为字母都是一样的才行，那么我们直接让string字符串按照顺序排列下来，一样的即可。很简单。但是这里时间复杂度最低也是NlogN，就算是快排。

第二种方法：Map：计数

{letter： Count}

Map -> Count {a:3, n:1.....}

时间复杂度来看，这只是在计数，所以是O(N)，如果再加上插入删除和查询，都是O(1)，所以合起来也是O(N)，所以这种方法略快于排序的算法。

代码：

```
class Solution:
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
#         dic1, dic2 = {}, {}
#         for item in s:
#             dic1[item] = dic1.get(item, 0) + 1
#         for item in t:
#             dic2[item] = dic2.get(item, 0) + 1
            
#         return dic1 == dic2

        dic1, dic2 = [0]*26, [0]*26 #一共也就26个英文字母
        for item in s:
            dic1[ord(item)-ord('a')] += 1 #计数中~
            
        for item in t:
            dic2[ord(item)-ord('a')] += 1
        
        return dic1 == dic2
```
 
 

.报数序列是指一个整数序列，按照其中的整数的顺序进行报数，得到下一个数。其前五项如下：

1.     1
2.     11
3.     21
4.     1211
5.     111221
1 被读作  "one 1"  ("一个一") , 即 11。
11 被读作 "two 1s" ("两个一"）, 即 21。
21 被读作 "one 2",  "one 1" （"一个二" ,  "一个一") , 即 1211。

给定一个正整数 n ，输出报数序列的第 n 项。

注意：整数顺序将表示为一个字符串。

示例 1:

输入: 1
输出: "1"
示例 2:

输入: 4
输出: "1211"
首先解释题目。我觉得问题就在于没有把题目解释清楚上。

1 读 one one

11 不读 one one one one,读 two one, 连着一起相同的数会先说数量再说值。

以上是基础。接下来看怎么得到下一项的结果。从题目所给出的示例4 ： 1211 到 5 : 111221。1211 第一位是1，所以读作 one one，也就是 1 1 .2读作one two, 
所以是12. 11连着读作two one, 所以是21.这所有加起来就是答案 111221。

思路：根据报数的特点，我们可以根据上一项的结果推导下一项。我们遍历上一项，辅以计数变量统计一下某些数字出现的次数。同时我们要不断保存上一项。
```
class Solution:  
    def countAndSay(self, n):  
        """ 
        :type n: int 
        :rtype: str 
        """  
        if n==1:#类似于斐波拉契数，后面的数跟前面的数有关  
            return '1'  
        if n==2:  
            return '11'
        #进行i=3时的循环时，它的上一项为'11'
        pre='11'
        
        #用for循环不断去计算逼近最后一次
        for i in range(3,n+1):  
            res=''#结果，每次报数都要初始化  
            cnt=1#计数变量
            
            length=len(pre)#遍历我们的上一项，所以记录它的长度
            for j in range(1,length):  
                if pre[j-1]==pre[j]:#相等则加一  
                    cnt+=1  
                else:
                    #一旦遇到不同的变量，就更新结果
                    res+=str(cnt)+pre[j-1]  
                    cnt=1#重置为1
            #把最后一项及它的数量加上
            res+=str(cnt)+pre[j]  
            pre=res#保存上一次的结果  
        return res
```



392. 判断子序列

给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

你可以认为 s 和 t 中仅包含英文小写字母。字符串 t 可能会很长（长度 ~= 500,000），而 s 是个短字符串（长度 <=100）。

字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

示例 1:
s = "abc", t = "ahbgdc"

返回 true.

示例 2:
s = "axc", t = "ahbgdc"

返回 false.

后续挑战 :

如果有大量输入的 S，称作S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？

**思路：**

这里又用到了双指针：

s: a  b  c

   |
   
   s_p


t: a  h  b  g  c  k

   |
   
  t_p
  
  
这里我们不断移动t_p指针，看t_p指向的元素是否和s_p指向的相等，如果不相等的话继续移动t_p，如果相等的话也一并移动s_p，直到t_p到达了t的边界。在这期间，
如果s_p已经到达了s的边界的话，就直接返回True。若整个循环结束，就是t遍历完都没有返回true的话，就说明不存在，返回false

代码：
```
class Solution:
    def isSubsequence(self, s, t):
        if s == None or t == None:  #判断字符串的是否为空
        return False
        
        len_s = len(s)    #长度获取
        len_t = len(t)    
        if len_t < len_s:   #判断长度的真实性
            return False
        if len_s == 0:
            return True
        j=0
        for i in range(len_t):    #若对于t串来讲，若和s相等，就继续移动
            if s[j] == t[i]:
                j+=1              
            if j == len_s:    #最终如果移动的次数和s的长度相等就返回True
                return True
        return False
```

python内置了find()函数可以快速定位字符的位置
```
class Solution:
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        for seq_s in s:
            s_index = t.find(seq_s)
            if s_index == -1:
                return False
            if s_index == len(t) - 1: #如果找到的匹配的s达到了t的长度
                t = str()             #字符串长度赋给t
            else:                   
                t = t[s_index+1:]       #若还没匹配完，从下一个开始继续
        return True
```




### 链表

**23.合并 k 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。**

示例:

输入:
[
  1->4->5,
  1->3->4,
  2->6
]
输出: 1->1->2->3->4->4->5->6

看到思路有heap，similar question有ugly number|| -> 这个是用heapq来解决的

那么就用heap吧？ heapsort

最简单的做法是只要每个list里面还有node，就把他们扔到minheap里面去，然后再把minheap pop，一个一个node连起来，听起来时间复杂度和空间复杂度都蛮高的。
直接merge必然是不好的，因为没有利用有序这个点，应该做的是每次取来一个，然后再把应该的下一个放入

写到这里瞬间明白和ugly number ii像的点了，甚至感觉跟find in sorted matrix ii也像

```py
class Solution:
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        from heapq import heappush, heappop
        node_pools = []
        lookup = collections.defaultdict(list)
        for head in lists:
            if head:
                heappush(node_pools, head.val)
                lookup[head.val].append(head)
        dummy = cur = ListNode(None)
        while node_pools:
            smallest_val = heappop(node_pools)
            smallest_node = lookup[smallest_val].pop(0)
            cur.next = smallest_node
            cur = cur.next
            if smallest_node.next:
                heappush(node_pools, smallest_node.next.val)
                lookup[smallest_node.next.val].append(smallest_node.next)
        return dummy.next
```


**206.反转链表**

反转一个单链表。

示例:

输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
进阶:
你可以迭代或递归地反转链表。你能否用两种方法解决这道题？

```py
class Solution:
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head  #为空或只有一个数直接返回
        pre = None
        cur = head #假设头结点是现在cur
        nxt = cur.next
        while nxt:
            cur.next = pre  #cur的下一个指向pre，也就是反过来
            pre = cur  # pre指向cur
            cur = nxt #将当前连接赋给下一个
            nxt = nxt.next
        cur.next = pre
        head = cur #头结点指向cur当前结点
        return cur
```

最近理解的一种方法

```py
def reverseList(head):
    if not head or not head.next:
        return head
    pre = None
    cur = head
    while cur: #当存在cur的时候就继续循环
        nxt = cur.next
	cur.next = pre
	pre = cur
	cur = nxt
    return pre 
```


**21.合并两个有序链表**

将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

示例：

输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4

思路：方法 1：递归

我们可以如下递归地定义在两个链表里的 merge 操作（忽略边界情况，比如空链表等）：
也就是说，两个链表头部较小的一个与剩下元素的 merge 操作结果合并。

算法

我们直接将以上递归过程建模，首先考虑边界情况。
特殊的，如果 l1 或者 l2 一开始就是 null ，那么没有任何操作需要合并，所以我们只需要返回非空链表。否则，我们要判断 l1 和 l2 哪一个的头元素更小，然后递归地决定下一个添加到结果里的值。如果两个链表都是空的，那么过程终止，所以递归过程最终一定会终止。


```py
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1,l2.next)
            return l2
```

想法

我们可以用迭代的方法来实现上述算法。我们假设 l1 元素严格比 l2元素少，我们可以将 l2 中的元素逐一插入 l1 中正确的位置。

算法

首先，我们设定一个哨兵节点 "prehead"(或者说虚拟结点dummyhead) ，这可以在最后让我们比较容易地返回合并后的链表。我们维护一个 cur 指针，我们需要做的是调整它的 next 指针。然后，我们重复以下过程，直到 l1 或者 l2 指向了 null ：如果 l1 当前位置的值小于等于 l2 ，我们就把 l1 的值接在 cur 节点的后面同时将 l1 指针往后移一个。否则，我们对 l2 做同样的操作。不管我们将哪一个元素接在了后面，我们都把 cur 向后移一个元素。

在循环终止的时候， l1 和 l2 至多有一个是非空的。由于输入的两个链表都是有序的，所以不管哪个链表是非空的，它包含的所有元素都比前面已经合并链表中的所有元素都要大。这意味着我们只需要简单地将非空链表接在合并链表的后面，并返回合并链表。

```py
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummyhead = ListNode(-1)
        cur = dummyhead
        while l1 and l2:
            if l1.val <= l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        if l1 is not None:
            cur.next = l1
        else:
            cur.next = l2
        return dummyhead.next
```

**2.回文链表**

请判断一个链表是否为回文链表。

示例 1:

输入: 1->2
输出: false
示例 2:

输入: 1->2->2->1
输出: true
进阶：
你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？
思路：最简单的思路就是把指针的下一个给在定义的列表中，迭代。在倒过来判断。

```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None or head.next is None:
            return True
        l = []
        p = head
        while p.next:
            l.append(p.val)
            p = p.next
        l.append(p.val)
        return l == l[::-1]
```

**203.移除链表元素**

删除链表中等于给定值 val 的所有节点。

示例:

输入: 1->2->6->3->4->5->6, val = 6
输出: 1->2->3->4->5

思路：为了避免我们删除掉头结点的情况，我们可以设立一个虚拟头结点。

```py
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummyhead = ListNode(-1)
        dummyhead.next = head
        cur = dummyhead
        while cur.next:
            if cur.next.val ==val:
                cur.next = cur.next.next
            else:
                cur = cur.next 
        return dummyhead.next
```

这次再来回顾一下链表的算法实现

**2.链表交换相邻元素-swap node**

思路：类似的，只要把相邻的两个结点互相指向即可。
```py
def swapPairs(self, head)
  pre,pre.next = self,head
  while pre.next and pre.next.next:
    a = pre.next
    b = a.next
    pre.next, b.next, a.next = b,a,b.next  
    pre = a
  return self.next
```

**3.环形链表**

给定一个链表，判断链表中是否有环。

进阶：
你能否不使用额外空间解决此题？

思路：这里有一个快慢指针的技巧，就是设置两个指针，一个走两步，另一个走一步，一开始都指向头结点，之后如果相遇则说明是环形，如果遇不到(不等于）则说明不是环形。

```py
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        fast = slow = head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False
```

**237. 删除链表中的节点**

请编写一个函数，使其可以删除某个链表中给定的（非末尾）节点，你将只被给定要求被删除的节点。

现有一个链表 -- head = [4,5,1,9]，它可以表示为:

示例 1:

输入: head = [4,5,1,9], node = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.

思路：注意边界条件，将node的下一个结点先赋值给node结点，然后跨过这个结点就可以，当不在node没有下一个结点的时候，我们把pre.next设置为空就好

```py
class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        if node is None:
            return
        while node.next:
            node.val = node.next.val
            pre = node
            node = node.next
        pre.next = None
```

**19. 删除链表的倒数第N个节点**

给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。

示例：

给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.

思路：技巧 dummy head 和双指针。

切记最后要返回dummy.next而不是head，因为可能删的就是head，例如：

输入链表为[1], n = 1, 应该返回None而不是[1]

p,q同时移动

始终记住只要是删除一个节点，就相当于跳过这个节点即：
p.next = p.next.next
指向后面那个结点

```py
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        assert n >= 0
        dummyhead = ListNode(-1)
        dummyhead.next = head
        p = q = dummyhead
        for i in range(0, n+1):
            assert q
            q = q.next
        while q is not None:
            p = p.next
            q = q.next
        p.next = p.next.next
        return dummyhead.next
```

**876. 链表的中间结点**
给定一个带有头结点 head 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。

思路：当用慢指针 slow 遍历列表时，让另一个指针 fast 的速度是它的两倍。

当 fast 到达列表的末尾时，slow 必然位于中间。

```py
class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        fast = slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```

### 堆/栈/队列

#### 1.有效的括号
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。

示例 1:

输入: "()"
输出: true
示例 2:

输入: "()[]{}"
输出: true
示例 3:

输入: "(]"
输出: false
示例 4:

输入: "([)]"
输出: false
示例 5:

输入: "{[]}"
输出: true

思路：这个可以用堆栈的思想去解决，先设立一个栈准备放东西，然后把括号组成一个键值对，然后往栈里面push值value，再判断进来的是否是括号的右边匹配部分，如果不是匹配的，就返回False，如果一开始就不在值里面，就相当于一开始就不是左括号，那一定是错误的False

```
class Solution:
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
       
        stack = []
        dict = {"]": "[", "}": "{", ")": "("}
        for char in s:
            if char in dict.values():
                stack.append(char)
            elif char in dict.keys():
                if stack == [] or dict[char] != stack.pop():
                    return False
            else:
                return False
        return stack == []
```


1.用栈实现队列
使用栈实现队列的下列操作：

push(x) -- 将一个元素放入队列的尾部。
pop() -- 从队列首部移除元素。
peek() -- 返回队列首部的元素。
empty() -- 返回队列是否为空。
示例:

MyQueue queue = new MyQueue();

queue.push(1);
queue.push(2);  
queue.peek();  // 返回 1
queue.pop();   // 返回 1
queue.empty(); // 返回 false
说明:

你只能使用标准的栈操作 -- 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。
假设所有操作都是有效的 （例如，一个空的队列不会调用 pop 或者 peek 操作）。

**思路**：1、用一个栈实现进队，另一个栈实现出队 
      2、需要进队的时候把元素压如stack1中，需要出队的时候把stack1中的元素全部弹出至stack2中 
      3、从stack2中出队，即可实现先进先出

```
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.stack1.append(x)
        

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        if self.stack2:
            return self.stack2.pop()
        else:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()

```




2.使用队列实现栈的下列操作：

push(x) -- 元素 x 入栈
pop() -- 移除栈顶元素
top() -- 获取栈顶元素
empty() -- 返回栈是否为空
注意:

你只能使用队列的基本操作-- 也就是 push to back, peek/pop from front, size, 和 is empty 这些操作是合法的。
你所使用的语言也许不支持队列。 你可以使用 list 或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。
你可以假设所有操作都是有效的（例如, 对一个空的栈不会调用 pop 或者 top 操作）。

**思路**：进栈：元素入队列A

出栈：判断如果队列A只有一个元素，则直接出队。否则，把队A中的元素出队并入队B，直到队A中只有一个元素，再直接出队。为了下一次继续操作，互换队A和队B。

复杂度分析：

第一种形式：如果以列表尾作为队尾，直接用 append 插入新元素，复杂度为O(1)。

再用pop去弹出队首，也就是列表第0个元素，弹出后插入到另一个队列中。第一次 pop，需要移动列表后面n-1个元素，第二次 pop，需要移动后面n-2个元素……直到最后只剩最后一个元素，直接出队。

复杂度：(n-1)+(n-2)+……+1=O(n^2)。

第二种形式：如果以列表首作为队尾，用 insert 插入新元素，需要移动后面的元素，复杂度则为O(n)。

再用pop去弹出队首，也就是列表最后一个元素，弹出后插入到另一个队列中。这样操作虽然弹出元素的复杂度为O(1)，但再插入另一个队列的复杂度则为O(n)，因为要连续弹出n-1个元素，则需要连续插入n-1个元素，最后的复杂度同样会是O(n^2)。

因此选择第一种形式。

而直接用python的一个列表实现栈，以列表尾为栈首，则出栈和进栈的复杂度都为O(1)。

实现：就以列表作为队列的底层实现，只要保证先进先出的约束就是队列。这里只实现进栈和出栈两个操作。

```
class Stock:
    def __init__(self):
        self.queueA=[]
        self.queueB=[]
    def push(self, node):
        self.queueA.append(node)
    def pop(self):
        if len(self.queueA)==0:
            return None
        while len(self.queueA)!=1:
            self.queueB.append(self.queueA.pop(0))
        self.queueA,self.queueB=self.queueB,self.queueA #交换是为了下一次的pop
        return self.queueB.pop()
```





### 二叉树

110.给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。

示例 1:

给定二叉树 [3,9,20,null,null,15,7]

    3
   / \
  9  20
    /  \
   15   7
返回 true 。

示例 2:

给定二叉树 [1,2,2,3,3,null,null,4,4]

       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
返回 false 。

思路：递归,判断左右子树最大高度差不超过1且左右子树均为平衡树

```py
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def getDepth(root):
            if not root:
                return 0
            return 1 + max(getDepth(root.left), getDepth(root.right)) #左右子树最大的深度，记住加一
    
        if not root:
            return True
        if abs(getDepth(root.left) - getDepth(root.right))>1:   #判断左右子树的最大深度差是否超过1
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)
```


513.给定一个二叉树，在树的最后一行找到最左边的值。

示例 1:

输入:

    2
   / \
  1   3

输出:
1
 

示例 2:

输入:

        1
       / \
      2   3
     /   / \
    4   5   6
       /
      7

输出:
7
 

注意: 您可以假设树（即给定的根节点）不为 NULL。

```py
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        result, queue = [], [root]
        while queue:
            temp = []
            result = queue[0].val       #取出每一层根的val
            for node in queue:      
                if node.left:               
                    temp.append(node.left)  #层次遍历
                if node.right:
                    temp.append(node.right)
            queue = temp
        return result
```





230.给定一个二叉搜索树，编写一个函数 kthSmallest 来查找其中第 k 个最小的元素。

说明：
你可以假设 k 总是有效的，1 ≤ k ≤ 二叉搜索树元素个数。

示例 1:

输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 1
示例 2:

输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 3

思路：有没有办法，在遍历的过程中就知道list的排序呢？啊！！！中序遍历。我们知道中序遍历的结果是一个有序的list，所以我们可以在中序遍历中设置提前停止。

```py
class Solution:
    def _inOrder(self,root,arr,k):
        if root:
            self._inOrder(root.left,arr,k)
            if len(arr) >= k:
                return 
            arr.append(root.val)
            self._inOrder(root.right,arr,k)

    def kthSmallest(self, root: TreeNode, k: int) -> int:
        res = []
        self._inOrder(root,res,k)
        return res[k-1]
```


208.实现一个 Trie (前缀树)，包含 insert, search, 和 startsWith 这三个操作。

示例:

Trie trie = new Trie();

trie.insert("apple");
trie.search("apple");   // 返回 true
trie.search("app");     // 返回 false
trie.startsWith("app"); // 返回 true
trie.insert("app");   
trie.search("app");     // 返回 true



```py
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        self.end_of_word = "#"
        
    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        node = self.root
        for char in word:
            node = node.setdefault(char,{})
        node[self.end_of_word] = self.end_of_word
            
    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return self.end_of_word in node

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
```





**235.给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。**

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]


示例 1:

输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。
```py
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root == None:
            return None
        if (p.val < root.val and q.val < root.val):
            return self.lowestCommonAncestor(root.left,p,q)
        if (p.val > root.val and q.val > root.val):
            return self.lowestCommonAncestor(root.right,p,q)
        return root
```




**104.给定一个二叉树，找出其最大深度。**

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例：
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 。

```py
def maxDepth(self,root):
    if not root:
        return 0
    return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

**111.给定一个二叉树，找出其最小深度。**

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

说明: 叶子节点是指没有子节点的节点。

示例:

给定二叉树 [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回它的最小深度  2.

```py
def minDepth(self,root):
    if root is None:
        return 0
    if root.left == None:
        return self.minDepth(root.right)+1
    if root.right == None:
        return self.minDepth(root.left)+1
    return min(self.minDepth(root.right),self.minDepth(root.left))+1
```


**100.给定两个二叉树，编写一个函数来检验它们是否相同。**

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

示例 1:

输入:       1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

输出: true
示例 2:

输入:      1          1
          /           \
         2             2

        [1,2],     [1,null,2]

输出: false

```py
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        if(p and not q) or (q and not p):
            return False
        return (p.val==q.val) and self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
```

**226.翻转一棵二叉树。**

示例：

输入：

     4
   /   \
  2     7
 / \   / \
1   3 6   9
输出：

     4
   /   \
  7     2
 / \   / \
9   6 3   1

```py
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None
        self.invertTree(root.left)
        self.invertTree(root.right)
        root.left,root.right = root.right,root.left
        return root
```

**94.给定一个二叉树，返回它的中序 遍历。**

示例:

输入: [1,null,2,3]
1

2
/
3

输出: [1,3,2]
进阶: 递归算法很简单，你可以通过迭代算法完成吗？

思路：就用非递归来写，先一股脑把左边一条线全部push到底（即走到最左边），然后node最终为None了就开始pop stack了，然后因为pop出来的每一个node都是自
己这棵树的root，所以看看它有没有右孩子，没有那肯定继续pop，有的话自然而然右孩子是下一个要被访问的节点。

```py
class Solution:
    def inorderTraversal(self, root: 'TreeNode') -> 'List[int]':
        if not root:
            return []
        stack = []
        node = root 
        res = []
        while node or stack:
            while node:             #把节点放入栈中
                stack.append(node)  
                node = node.left    #一直往左边找
            node = stack.pop()      #pop掉左边
            res.append(node.val)    #作为结果
            node = node.right          #再遍历右边子树
        return res
```

**145.二叉树的后序遍历**

给定一个二叉树，返回它的 后序 遍历。

示例:

输入: [1,null,2,3]  
   1
    \
     2
    /
   3 

输出: [3,2,1]
进阶: 递归算法很简单，你可以通过迭代算法完成吗？


我们实际上可以模拟栈的操作。对于这个问题，实际上在计算机中是这样处理的。我们首先将打印node1.val、访问node1的right和访问node1的left压入栈中。

stack : cout1   go-1-R   go-1-L

然后弹出访问node1的left，我们发现它是空，所以什么都不操作。接着我们访问node1的right，

stack : cout1   cout2   go-2-R   go-2-L   

然后弹出go-2-L，我们接着将打印node3.val、访问node3的right和访问node3的left压入栈中。

stack : cout1   cout2   go-2-R   cout3   go-3-R   go-3-L 

接着就是弹出这些指令就可以了。


```py
class Solution:
    def postorderTraversal(self, root: 'TreeNode') -> 'List[int]':
	res = []
	if root is None:
	    return []
	stack = [root]
	while stack:
	    res = stack.pop()
	    if root.left:
	        stack.append(root.left)
	    if root.right:
	        stack.append(root.right)
	return res[::-1]
```

**102.给定一个二叉树，返回其按层次遍历的节点值。 （即逐层地，从左到右访问所有节点）。**

例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [9,20],
  [15,7]
]

思路：

该问题需要用到队列

建立一个queue

先把根节点放进去，这时候找根节点的左右两个子节点

去掉根节点，此时queue里的元素就是下一层的所有节点

用for循环遍历，将结果存到一个一维向量里

遍历完之后再把这个一维向量存到二维向量里

以此类推，可以完成层序遍历


![层次遍历](https://bucket-1257126549.cos.ap-guangzhou.myqcloud.com/20181112084159.gif)

```py
class Solution:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        result = []
        queue = collections.deque()
        queue.append(root)
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            result.append(current_level)
        return result
```

### 动态规划

**爬楼梯**
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

注意：给定 n 是一个正整数。

示例 1：

输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶
示例 2：

输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
3.  1 阶 + 1 阶 + 1 阶
4.  1 阶 + 2 阶
5.  2 阶 + 1 阶

**思路：**
爬楼梯算是DP的经典题目，递归+记忆化，也就是递推，我们需要定义号状态，还有状态的转移方程。最后爬的步数还是得看之前的，即依赖之前的步骤。

1.反着考虑，有几种方案到第i阶楼梯，答案是2种：

第i-1阶楼梯经过一步
第i-2阶楼梯经过两步
假设count(i)表示到第i阶楼梯方案的个数，则count(i) = count(i-1) + count(i-2) 
第一阶是1种，第二阶是2种。代码如下：

```
class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = [1,2]   #一次就只能走这两步
        for i in range(2,n):
            count.append(count[i-1]+count[i-2])	#不停地把后面的台阶的结束放到count里面
        return count[n-1]
```
但是太慢了。。。这里起码O(n!)

2.我们想到可以转化为fibonaqi问题。假设一共有10阶楼梯，每步可以爬1步或者2步，那么你爬到10阶一共有两种方法，从8阶爬2步，或从9阶爬1步，那么爬到9阶也是这样，那这就是一共基本的斐波那契数列。
dp[i] = dp[i-1] + dp[i-2]
i-1的时候跳一步可以到达i
i-2的时候跳一步是i-1，这个变成dp[i-1]的子问题了,直接跳两步可以到达i

```
class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [1 for i in range(n+1)]   #状态的定义
        for i in range(2,n+1):
            dp[i] = dp[i-1]+dp[i-2]	#状态转移方程
        return dp[n]
```
这里应该是O(n^2)
3.还有一种更快速的，列表初始化好，然后再用fibonaqi数列转移方程。

```
class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        condition = [0]*(n+1)		#牛逼的初始化列表
        condition[0] = 1
        condition[1] = 1
        for i in range(2,n+1):
            condition[i] = condition[i-1]+condition[i-2]   #依然还是状态转移fibonaqo
        return condition[n]
```
这里列表初始化只用来O(1)，最后复杂度为O(n)




## 2.review
[机器学习无监督学习算法——聚类](https://en.wikipedia.org/wiki/Cluster_analysis)

无监督学习中，聚类的算法有超过100种，但是目的都是从庞大的样本中选出有代表性的加以学习，或者选出很棒的特征来。
常用的有k-means，用于簇是密集的情况。类内紧凑，类间独立。
还有DBSCAN（具有噪声的基于密度的聚类），将具有足够密度的区域划分为簇，并在具有噪声的空间数据库中发现任意形状的簇。他将簇定义为密度相连的点的最大集合。
还有局部密度聚类，层次聚类和谱聚类。

## 3.tips
由于还在学校念书，就把科研或者遇到的问题记录下来。
一个循环队列的问题:
我们之所以把静态队列归为循环队列，是因为用数组实现的队列，如果仅仅是像普通的链式队列的话会导致浪费内存空间。
![循环队列](https://gss1.bdstatic.com/-vo3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=2d7e279f259759ee5e5d6899d3922873/5d6034a85edf8db1ee973ff60a23dd54574e74e2.jpg)

一共有两个参数：
front和rear，这里简称f和r
内存中增加一个内容，r就往下一个移动，r是指向最后一个元素的下一个元素，若队列为空，f和r一定相等。
这样的话f和r就不会冲突。
(r+1)%长度 == f的话就说明整个队列满了。r==f就代表队列是空的。

## 4.share
这次分享一下读书心得，最近读《小王子》，这里[](https://book.douban.com/review/1000104/)
一切虽然有些讽刺大人们的物质功利世界，但是，我们又何尝不是活成了不想成为的样子呢？
一个小孩说，有一种车他最喜欢。于是大人们纷纷开始猜测，奔驰，大卡，云云。结果小孩说，他最喜欢垃圾车。
因为垃圾车来的时候，都有音乐。
