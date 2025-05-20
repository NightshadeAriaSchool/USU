import random

ADJECTIVE_CHANCE = 0.2
CALLING_CHANCE = 0.2

class AgentName:
    adjectives = ['Smart', 'Quick', 'Clever', 'Lazy', 'Brave', 'Sneaky', 'Mystic', 'Electric', 'Shy']
    animals = ['Fox', 'Cat', 'Wolf', 'Dragon', 'Bunny', 'Tiger', 'Raven', 'Panda']
    callings = ['the Sly', 'the Swift', 'the Wise', 'the Fierce', 'the Shadow', 'the Tiny', 'the Arcane', 'the Cuddly']
    names = ['Kitsu', 'Mochi', 'Luna', 'Yuki', 'Hana', 'Tora', 'Neko', 'Kuro', 'Mimi']
    ranks = ["Level", "Version", "Stage", "Phase", "Mark", "Generation", "Ξ"]

    def __init__(self, adjective:str, animal:str, calling:str, name:str, rank:str, epoch:int, parent:str=None):
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
    
    def __str__(self):
        return self.full_name
    
    def increment(self) -> "AgentName":
        return AgentName(self.adjective, self.animal, self.calling, self.name, self.rank, self.epoch + 1, self.parent)
    
    def fork(self) -> "AgentName":
        adjective = random.choice(AgentName.adjectives) if random.random() < ADJECTIVE_CHANCE else self.adjective
        calling = random.choice(AgentName.callings) if random.random() < CALLING_CHANCE else self.calling
        name = random.choice(AgentName.names)
        return AgentName(adjective, self.animal, calling, name, self.rank, self.epoch, self.name)
    
    @staticmethod
    def random(epoch:int=1):
        adjective = random.choice(AgentName.adjectives)
        animal = random.choice(AgentName.animals)
        calling = random.choice(AgentName.callings)
        name = random.choice(AgentName.names)
        rank = random.choice(AgentName.ranks)
        return AgentName(adjective, animal, calling, name, rank, epoch)

def new_name():
    return AgentName.random(1)