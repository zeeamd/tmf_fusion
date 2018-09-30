
# # Test Jenkins
#
val1 = 100.99
val2 = 76.15

# Adding the two given numbers
sum = float(val1) + float(val2)

# Displaying the addition result
print("The sum of given numbers is: ", sum)
if sum == 117.14:
    print("test pass")
else:
    print("TEST FILE RUNNING")

# import random
# try:
#     import unittest2 as unittest
# except ImportError:
#     import unittest
#
# class SimpleTest(unittest.TestCase):
#     @unittest.skip("demonstrating skipping")
#     def test_skipped(self):
#         self.fail("shouldn't happen")
#
#     def test_pass(self):
#         self.assertEqual(10, 7 + 3)
#
#     def test_fail(self):
#         self.assertEqual(11, 7 + 3)