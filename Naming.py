import random

ADJECTIVE_CHANCE = 0.2
CALLING_CHANCE = 0.2

_id = 0
_used_names = []

class AgentName:
    adjectives = ['Smart', 'Quick', 'Clever', 'Lazy', 'Brave', 'Sneaky', 'Mystic', 'Electric', 'Shy']
    animals = ['Fox', 'Cat', 'Wolf', 'Dragon', 'Bunny', 'Tiger', 'Raven', 'Panda']
    callings = ['the Sly', 'the Swift', 'the Wise', 'the Fierce', 'the Shadow', 'the Tiny', 'the Arcane', 'the Cuddly']
    names = ['Kitsu', 'Mochi', 'Luna', 'Yuki', 'Hana', 'Tora', 'Neko', 'Kuro', 'Mimi', 'Aria']
    ranks = ["Level", "Version", "Stage", "Phase", "Mark", "Generation", "Ξ"]

    def __init__(self, adjective:str, animal:str, calling:str, name:str, rank:str, epoch:int, id:int = _id, parent: 'AgentName' = None):
        self.id = id
        self.adjective = adjective
        self.animal = animal
        self.calling = calling
        self.name = name
        self.rank = rank
        self.epoch = epoch
        self.parent = parent
    
    @property
    def full_name(self):
        rs = ""
        if self.epoch >= 10: rs += self.adjective + " "
        rs += self.animal
        if self.epoch >= 100: rs += " " + self.calling
        rs += " " + self.name
        if self.parent is not None: rs += " of " + self.parent
        rs += " " + self.rank + " " + str(self.epoch)
        return rs
    
    @property
    def persona_name(self):
        rs = ""
        if self.epoch >= 10: rs += self.adjective + " "
        rs += self.animal
        if self.epoch >= 100: rs += " " + self.calling
        rs += " " + self.name
        return rs
    
    @property
    def color(self) -> float:
        if self.id < 3:
            return self.id
        
        count = 6
        increment = 1
        while self.id >= count:
            count += count
            increment /= 2
        return (self.id - count / 2) * increment + increment / 2
        
    def __str__(self):
        return self.full_name
    
    def increment(self) -> "AgentName":
        return AgentName(self.adjective, self.animal, self.calling, self.name, self.rank, self.epoch + 1, self.id)
    
    def fork(self) -> "AgentName":
        global _used_names
        adjective = random.choice(AgentName.adjectives) if random.random() < ADJECTIVE_CHANCE else self.adjective
        calling = random.choice(AgentName.callings) if random.random() < CALLING_CHANCE else self.calling
        name = random.choice([name for name in AgentName.names if name not in _used_names])
        _used_names.append(name)
        if len(_used_names) == len(AgentName.names): _used_names = []
        global _id; id = _id; id += 1
        return AgentName(adjective, self.animal, calling, name, self.rank, self.epoch, id)
    
    @staticmethod
    def random(epoch:int=1):
        adjective = random.choice(AgentName.adjectives)
        animal = random.choice(AgentName.animals)
        calling = random.choice(AgentName.callings)
        global _used_names
        name = random.choice([name for name in AgentName.names if name not in _used_names])
        _used_names.append(name)
        if len(_used_names) == len(AgentName.names): _used_names = []
        rank = random.choice(AgentName.ranks)
        return AgentName(adjective, animal, calling, name, rank, epoch)

def new_name():
    name = AgentName.random(1)
    global _id; _id += 1
    return name