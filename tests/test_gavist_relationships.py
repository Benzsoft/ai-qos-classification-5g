import unittest
import pandas as pd
import numpy as np
from research.gavist_relationships import describe,relationship,shingles
class Tests(unittest.TestCase):
 def frame(self):
  return pd.DataFrame({'Time':pd.date_range('2024-01-01',periods=150,freq='s'),'Source':['a']*150,'Destination':['b']*150,'Protocol':['UDP']*150,'Length':np.arange(150)**2+1})
 def test_transformed_copy(self):
  a=self.frame();b=a.copy();b.Time+=pd.Timedelta(days=2);b['ARTT']=999
  r=relationship(describe(a),describe(b))
  self.assertIn('shared_time_shifted_record_fingerprints',r['reasons'])
  self.assertIn('shared_variable_byte_sequence_shingles',r['reasons'])
  self.assertEqual(r['shared_absolute_records'],0)
 def test_endpoints_alone_not_grouped(self):
  a=self.frame();b=a.copy();b.Time+=pd.Timedelta(days=2);b.Length+=123456
  self.assertEqual(relationship(describe(a),describe(b))['reasons'],[])
 def test_constant_windows_excluded(self):self.assertEqual(shingles(np.ones(100,dtype=int)),set())
 def test_invalid_rejected(self):
  a=self.frame();a.loc[0,'Length']=-1
  with self.assertRaises(ValueError):describe(a)
if __name__=='__main__':unittest.main()
