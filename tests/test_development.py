import unittest
import numpy as np
from research.train_development import preprocess

class PreprocessingTests(unittest.TestCase):
    def test_training_only_and_prefix(self):
        train={'X':np.ones((3,30,3),dtype=np.float32),'protocols':np.full((3,30),'train'),'endpoints':np.full((3,2),'known'),'y':np.array(['a','b','c'])}
        test={'X':np.full((3,30,3),100,dtype=np.float32),'protocols':np.full((3,30),'unseen'),'endpoints':np.full((3,2),'unseen'),'y':np.array(['a','b','c'])}
        prepared,state=preprocess({'train':train,'test':test},5,False)
        self.assertEqual(prepared['train']['seq'].shape,(3,5,3))
        np.testing.assert_allclose(state['scaler'].mean_,[1,1,1])
        self.assertTrue((prepared['test']['endpoints']==0).all())
        self.assertNotIn('unseen',state['protocolmap'])
        self.assertEqual(train['X'].shape,(3,30,3))
        p,_=preprocess({'train':train,'test':test},10,True)
        self.assertTrue((p['test']['seq'][:,:,3:]==0).all())

if __name__=='__main__':unittest.main()
