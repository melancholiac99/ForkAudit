import unittest
from analyze import first,box,VERIFY,REFLECT

class AnswerRules(unittest.TestCase):
    def test_first(self):
        for text,val in [('The answer is 12. Now reconsider.','12'),('my answer should be $3/4$.', '3/4'),
                         (r'The answer is 2. Actually, \boxed{3}', '2'),
                         (r'First \boxed{\frac{1}{2}}. The answer is 3.',r'\frac{1}{2}')]:
            self.assertEqual(first(text,box(text))[1],val)
    def test_unmarked_arithmetic(self):
        self.assertIsNone(first('Compute 1+3=4. Next convert units.',None))
        self.assertIsNone(first('The answer is 2/3+4.',None))
        self.assertIsNone(box(r'Not complete: \boxed{\frac{1}{2}'))
if __name__=='__main__':unittest.main()
