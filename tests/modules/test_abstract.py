import unittest
from yann.modules.abstract import module as mod

class TestAbstractModule(unittest.TestCase):
    def test_abstract(self):
        self.verbose = 3
        self.id ="AbsMod"
        self.type = "resultor"
        self.mod = mod(id=self.id,type=self.type,verbose=self.verbose)
        self.assertTrue(self.mod.id,self.id)
        self.assertTrue(self.mod.type,self.type)
        print()
