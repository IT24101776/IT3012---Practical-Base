# agent.py

from collections import deque
import heapq
import random

class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """A purely reactive agent: action depends only on the current percept.
    No __init__, no stored state — every decision is a fresh IF-THEN lookup."""

    def sense_and_act(self, percept: dict) -> str:
        if percept['wall_ahead']:
            return 'Left'          # condition-action rule: wall -> turn/step left
        elif percept['food_here']:
            return 'Up'            # condition-action rule: food -> move onto/through it
        else:
            return 'Right'         # default: keep exploring


class ModelBasedAgent:
    """A memory-based agent that tracks visited locations and avoids repeating a blocked path."""

    def __init__(self):
        self.believed_pos = (0, 0)
        self.visited_cells = {self.believed_pos}
        self.last_action = None
    
        self._turn_order = ['Left', 'Down', 'Right', 'Up']

    def _neighbor(self, action):
        x, y = self.believed_pos
        if action == 'Up':
            return (x, y + 1)
        if action == 'Down':
            return (x, y - 1)
        if action == 'Left':
            return (x - 1, y)
        if action == 'Right':
            return (x + 1, y)
        return self.believed_pos

    def sense_and_act(self, percept: dict) -> str:
        # 1. UPDATE STATE (Transition & Sensor Model) ------------------------
        # If our last move plausibly succeeded (the world isn't now telling
        # us there's a wall right in front of us), fold it into our internal
        # map: shift our believed position and mark the new cell visited.
        if self.last_action is not None and not percept.get('wall_ahead', False):
            self.believed_pos = self._neighbor(self.last_action)
            self.visited_cells.add(self.believed_pos)
 
        # 2. IF-THEN RULES THAT QUERY MEMORY ----------------------------------
        if percept.get('wall_ahead', False):
            # IF wall_ahead AND <candidate direction already tried or leads
            # somewhere we've already been> THEN skip it and try the next
            # candidate. This is the memory check that breaks the loop.
            action = None
            for candidate in self._turn_order:
                already_tried = (candidate == self.last_action)
                leads_to_visited = self._neighbor(candidate) in self.visited_cells
                if not already_tried and not leads_to_visited:
                    action = candidate
                    break
            if action is None:
                # Every option has been tried or leads somewhere visited —
                # fall back to anything that isn't literally the last action,
                # so we never repeat ourselves twice in a row.
                action = next(a for a in self._turn_order if a != self.last_action)
 
        elif percept.get('food_here', False):
            action = 'Up'  # move onto/through the food cell
 
        else:
            # Open path ahead: keep going the way we were already heading,
            # or start moving if this is the very first decision.
            action = self.last_action if self.last_action else 'Up'
 
        self.last_action = action
        return action

class SearchAgent:
    """A problem-solving agent that searches a grid for a goal."""

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

    def sense_and_act(self, percept):
        if not self.plan:
            current_position = tuple(percept['agent_pos'])
            food_positions = [tuple(food) for food in percept.get('all_food', [])]

            if not food_positions:
                return 'Up'

            goal_position = min(
                food_positions,
                key=lambda food: abs(food[0] - current_position[0])
                + abs(food[1] - current_position[1]),
            )
            walls = {tuple(wall) for wall in percept.get('walls', [])}
            grid_size = percept['grid_size']

            search_methods = {
                'BFS': self.bfs_search,
                'DFS': self.dfs_search,
                'UCS': self.ucs_search,
            }
            if self.active_algo not in search_methods:
                raise ValueError(f'Unknown search algorithm: {self.active_algo}')

            self.plan = search_methods[self.active_algo](
                current_position, goal_position, walls, grid_size
            ) or []

            if not self.plan:
                return 'Up'

        return self.plan.pop(0)

    def _neighbors(self, position, walls, grid_size):
        directions = {
            'Up': (0, 1),
            'Down': (0, -1),
            'Left': (-1, 0),
            'Right': (1, 0),
        }

        for action, (dx, dy) in directions.items():
            next_position = (position[0] + dx, position[1] + dy)
            inside_grid = (
                0 <= next_position[0] < grid_size[0]
                and 0 <= next_position[1] < grid_size[1]
            )
            if inside_grid and next_position not in walls:
                yield action, next_position

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        queue = deque([(start_pos, [])])
        reached = {start_pos}

        while queue:
            current_position, path = queue.popleft()
            if current_position == goal_pos:
                return path

            for action, next_position in self._neighbors(
                current_position, walls, grid_size
            ):
                if next_position not in reached:
                    reached.add(next_position)
                    queue.append((next_position, path + [action]))

        return None

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        stack = [(start_pos, [])]
        reached = {start_pos}

        while stack:
            current_position, path = stack.pop()
            if current_position == goal_pos:
                return path

            neighbors = list(self._neighbors(current_position, walls, grid_size))
            for action, next_position in reversed(neighbors):
                if next_position not in reached:
                    reached.add(next_position)
                    stack.append((next_position, path + [action]))

        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        frontier = [(0, 0, start_pos, [])]
        reached = {start_pos: 0}
        sequence = 1

        while frontier:
            path_cost, _, current_position, path = heapq.heappop(frontier)
            if current_position == goal_pos:
                return path

            for action, next_position in self._neighbors(
                current_position, walls, grid_size
            ):
                new_cost = path_cost + 1
                if next_position not in reached or new_cost < reached[next_position]:
                    reached[next_position] = new_cost
                    heapq.heappush(
                        frontier,
                        (new_cost, sequence, next_position, path + [action]),
                    )
                    sequence += 1

        return None
