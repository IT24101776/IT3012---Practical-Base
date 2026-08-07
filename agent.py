# agent.py
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
