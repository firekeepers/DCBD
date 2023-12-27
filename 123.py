import collections

def maximumSafenessFactor(grid):
    n = len(grid)
    m = len(grid[0])

    # 定义方向数组
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    # 以所有危险点为起点进行BFS
    dis = [[-1] * m for _ in range(n)]
    q = collections.deque()

    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1:
                q.append((i, j))
                dis[i][j] = 0

    while q:
        x, y = q.popleft()
        for dx, dy in directions:
            i, j = x + dx, y + dy
            if 0 <= i < n and 0 <= j < m and dis[i][j] == -1:
                q.append((i, j))
                dis[i][j] = dis[x][y] + 1

    # 通过一次BFS，检查能否只经过安全系数大于等于lim的格子，从左上角走到右下角
    def check(lim):
        visited = [[False] * m for _ in range(n)]
        q = collections.deque()
        q.append((0, 0))
        visited[0][0] = True

        while q:
            i, j = q.popleft()
            for dx, dy in directions:
                ii, jj = i + dx, j + dy
                if 0 <= ii < n and 0 <= jj < m and dis[ii][jj] >= lim and not visited[ii][jj]:
                    q.append((ii, jj))
                    visited[ii][jj] = True

        return visited[n - 1][m - 1]

    # 二分答案
    head, tail = 0, min(dis[0][0], dis[n - 1][m - 1])

    while head < tail:
        mid = (head + tail + 1) // 2
        if check(mid):
            head = mid
        else:
            tail = mid - 1

    return head

grid = [
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 0]
]

result = maximumSafenessFactor(grid)
print(result)
