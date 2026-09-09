import unittest
import pandas as pd
from research.gavist_verified_split import records
class VerificationTests(unittest.TestCase):
 def test_multiplicity(self):
  d=pd.DataFrame({'Time':['2024-01-01']*2,'Source':['a']*2,'Destination':['b']*2,'Protocol':['UDP']*2,'Length':[100,100]})
  a=records(d);b=records(d.iloc[:1]);self.assertEqual(sum((a-b).values()),1)
 def test_derived_fields_ignored(self):
  d=pd.DataFrame({'Time':['2024-01-01'],'Source':['a'],'Destination':['b'],'Protocol':['UDP'],'Length':[100]})
  b=d.copy();b['ARTT']=999;b['City']='other';self.assertEqual(records(d),records(b))
