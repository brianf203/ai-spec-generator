import unittest
from partition import partition, partition_first, partition_mid, three_way_partition
class Test(unittest.TestCase):
    def test_partition(self): lst=[3,1,2]; p=partition(lst,0,2); self.assertIn(p,[0,1,2])
    def test_first(self): lst=[3,1,2]; p=partition_first(lst,0,2); self.assertIn(p,[0,1,2])
    def test_mid(self): lst=[3,1,2]; p=partition_mid(lst,0,2); self.assertIn(p,[0,1,2])
    def test_three_way(self): lst=[2,1,2,3,2]; lt,gt=three_way_partition(lst,0,4); self.assertLessEqual(lt, gt)