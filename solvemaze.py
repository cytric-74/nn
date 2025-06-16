class MazeSolver:
    def __init__(self, maze):
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0]) if self.height > 0 else 0

    class PriorityQueue:
        def __init__(self):
            self.elements = []
        
        def empty(self):
            return len(self.elements) == 0
        
        def put(self, item, priority):
            self.elements.append((priority, item))
            self.elements.sort(key=lambda x: x[0])
        
        def get(self):
            return self.elements.pop(0)[1]

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.height and 0 <= ny < self.width and self.maze[nx][ny] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def solve(self, start, end):
        frontier = self.PriorityQueue()
        frontier.put(start, 0)
        
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while not frontier.empty():
            current = frontier.get()
            
            if current == end:
                break
            
            for neighbor in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, end)
                    frontier.put(neighbor, priority)
                    came_from[neighbor] = current
        
        # reconstruct path
        current = end
        path = []
        while current != start:
            path.append(current)
            current = came_from.get(current, None)
            if current is None:
                return None  # dead end
        path.append(start)
        path.reverse()
        return path

    def visualize(self, path=None):
        for i in range(self.height):
            row = []
            for j in range(self.width):
                if path and (i, j) in path:
                    if (i, j) == path[0]:
                        row.append('S')
                    elif (i, j) == path[-1]:
                        row.append('E')
                    else:
                        row.append('•')
                else:
                    row.append(' ' if self.maze[i][j] == 0 else '█')
            print(' '.join(row))


maze = [
    [0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0]
]

solver = MazeSolver(maze)
solution = solver.solve((0, 0), (7, 7))

if solution:
    print(f"Path found with {len(solution)} steps:")
    solver.visualize(solution)
else:
    print("No solution exists")