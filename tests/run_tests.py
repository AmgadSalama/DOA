"""
Test Runner Script
==================

Script to run all unit tests for the DOA methods repository.
"""

import unittest
import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_all_tests():
    """
    Discover and run all tests in the tests directory.
    """
    print("DOA Methods Test Suite")
    print("=" * 50)
    
    # Discover all test modules
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        failfast=False
    )
    
    print(f"Running tests from: {start_dir}")
    print(f"Test pattern: test_*.py")
    print("-" * 50)
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    # Return success status
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        
    return success


def run_specific_test(test_module):
    """
    Run tests from a specific module.
    
    Parameters
    ----------
    test_module : str
        Name of the test module (without .py extension)
    """
    print(f"Running tests from: {test_module}")
    print("=" * 50)
    
    # Import and run specific test module
    try:
        module = __import__(test_module)
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except ImportError as e:
        print(f"Error importing {test_module}: {e}")
        return False


def run_quick_test():
    """
    Run a quick smoke test to verify basic functionality.
    """
    print("DOA Methods Quick Test")
    print("=" * 30)
    
    try:
        # Test basic imports
        from doa_methods.array_processing import UniformLinearArray, SignalModel
        from doa_methods.classical import ConventionalBeamforming
        from doa_methods.subspace import MUSIC
        
        print("✅ All imports successful")
        
        # Test basic functionality
        array = UniformLinearArray(M=8, d=0.5)
        signal_model = SignalModel(array)
        
        # Generate test data
        X, S, N = signal_model.generate_signals(
            doas=[0], N_snapshots=50, snr_db=10, seed=42)
        
        print("✅ Signal generation successful")
        
        # Test classical method
        cbf = ConventionalBeamforming(array)
        doas_cbf = cbf.estimate(X, K=1)
        
        print("✅ Conventional beamforming successful")
        
        # Test subspace method
        music = MUSIC(array)
        doas_music = music.estimate(X, K=1)
        
        print("✅ MUSIC estimation successful")
        
        print("\n🎉 Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'quick':
            success = run_quick_test()
        elif sys.argv[1] in ['test_array_processing', 'test_classical_methods', 'test_subspace_methods']:
            success = run_specific_test(sys.argv[1])
        else:
            print(f"Unknown test option: {sys.argv[1]}")
            print("Available options: quick, test_array_processing, test_classical_methods, test_subspace_methods")
            success = False
    else:
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)