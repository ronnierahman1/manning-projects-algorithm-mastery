#!/usr/bin/env python3
"""
Comprehensive Test Runner and Demo for Enhanced Greedy Priority Algorithm

This script demonstrates all the enhancements made to the GreedyPriority system
and provides comprehensive testing and benchmarking capabilities.

Usage:
    python test_runner.py [--benchmark] [--verbose] [--quick]
"""

import argparse
import time
import sys, os
import json
from typing import Dict, Any, List
import traceback

# Add the root directory to sys.path for importing main app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our enhanced modules
from milestones.greedy_priority import GreedyPriority, Priority, QueryMetrics
from test_utilities import (
    QueryGenerator, PerformanceProfiler, ValidationHelper, 
    RealWorldScenarios, TestQuery
)


class TestRunner:
    """Comprehensive test runner for the GreedyPriority system."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.prioritizer = GreedyPriority()
        self.generator = QueryGenerator()
        self.profiler = PerformanceProfiler()
        self.validator = ValidationHelper()
        self.scenarios = RealWorldScenarios()
        self.results = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        if self.verbose or level in ["ERROR", "RESULT"]:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def run_basic_functionality_tests(self) -> Dict[str, Any]:
        """Run basic functionality tests."""
        self.log("Running basic functionality tests...")
        
        results = {
            'name': 'Basic Functionality',
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        test_cases = [
            {
                'name': 'Critical Priority Detection',
                'query': 'Emergency: Server down immediately!',
                'expected': Priority.CRITICAL
            },
            {
                'name': 'High Priority Detection', 
                'query': 'Need help with important deadline',
                'expected': Priority.HIGH
            },
            {
                'name': 'Medium Priority Detection',
                'query': 'What is machine learning?',
                'expected': Priority.MEDIUM
            },
            {
                'name': 'Low Priority Detection',
                'query': 'Hello, how are you today?',
                'expected': Priority.LOW
            },
            {
                'name': 'Empty Query Handling',
                'query': '',
                'expected': Priority.LOW
            },
            {
                'name': 'Long Complex Query',
                'query': 'I need a comprehensive step by step detailed explanation of how to implement a scalable microservices architecture with proper security, authentication, and performance optimization for a high-traffic production environment.',
                'expected': Priority.HIGH
            }
        ]
        
        for test_case in test_cases:
            try:
                actual = self.prioritizer.get_priority(test_case['query'])
                if actual == test_case['expected']:
                    results['passed'] += 1
                    self.log(f"✓ {test_case['name']}: {actual.name}")
                else:
                    results['failed'] += 1
                    error_msg = f"Expected {test_case['expected'].name}, got {actual.name}"
                    results['errors'].append(f"{test_case['name']}: {error_msg}")
                    self.log(f"✗ {test_case['name']}: {error_msg}", "ERROR")
                    
            except Exception as e:
                results['failed'] += 1
                error_msg = f"Exception: {str(e)}"
                results['errors'].append(f"{test_case['name']}: {error_msg}")
                self.log(f"✗ {test_case['name']}: {error_msg}", "ERROR")
        
        return results
    
    def run_pattern_recognition_tests(self) -> Dict[str, Any]:
        """Test regex pattern recognition capabilities."""
        self.log("Running pattern recognition tests...")
        
        results = {
            'name': 'Pattern Recognition',
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        pattern_tests = [
            {
                'name': 'System Failure Pattern',
                'query': "The application can't work properly after the update",
                'expected_min': Priority.CRITICAL,
                'expected_max': Priority.HIGH
            },
            {
                'name': 'Deadline Pattern',
                'query': "Need help with deployment due tomorrow",
                'expected_min': Priority.HIGH,
                'expected_max': Priority.MEDIUM
            },
            {
                'name': 'Question Pattern',
                'query': "Can you help me understand this implementation?",
                'expected_min': Priority.MEDIUM,
                'expected_max': Priority.LOW
            },
            {
                'name': 'Multiple Question Marks',
                'query': "What is this??? How does it work???",
                'expected_min': Priority.MEDIUM,
                'expected_max': Priority.LOW
            }
        ]
        
        for test in pattern_tests:
            try:
                actual = self.prioritizer.get_priority(test['query'])
                if test['expected_min'].value <= actual.value <= test['expected_max'].value:
                    results['passed'] += 1
                    self.log(f"✓ {test['name']}: {actual.name}")
                else:
                    results['failed'] += 1
                    error_msg = f"Priority {actual.name} not in expected range {test['expected_min'].name}-{test['expected_max'].name}"
                    results['errors'].append(f"{test['name']}: {error_msg}")
                    self.log(f"✗ {test['name']}: {error_msg}", "ERROR")
                    
            except Exception as e:
                results['failed'] += 1
                error_msg = f"Exception: {str(e)}"
                results['errors'].append(f"{test['name']}: {error_msg}")
                self.log(f"✗ {test['name']}: {error_msg}", "ERROR")
        
        return results
    
    def run_sorting_and_queue_tests(self) -> Dict[str, Any]:
        """Test sorting and priority queue functionality."""
        self.log("Running sorting and queue tests...")
        
        results = {
            'name': 'Sorting and Queue',
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Test sorting
        test_queries = [
            "Hello world",                    # LOW
            "Critical system failure",        # CRITICAL
            "How does this work?",           # MEDIUM  
            "Need urgent help",              # HIGH
            "Thanks for everything"          # LOW
        ]
        
        try:
            sorted_queries = self.prioritizer.sort_queries_by_priority(test_queries)
            priorities = [priority for priority, _ in sorted_queries]
            
            if priorities == sorted(priorities):
                results['passed'] += 1
                self.log("✓ Query sorting works correctly")
            else:
                results['failed'] += 1
                error_msg = f"Sorting failed: {[p.name for p in priorities]}"
                results['errors'].append(f"Query Sorting: {error_msg}")
                self.log(f"✗ Query Sorting: {error_msg}", "ERROR")
                
        except Exception as e:
            results['failed'] += 1
            error_msg = f"Sorting exception: {str(e)}"
            results['errors'].append(f"Query Sorting: {error_msg}")
            self.log(f"✗ Query Sorting: {error_msg}", "ERROR")
        
        # Test priority queue
        try:
            # Clear any existing queue
            self.prioritizer.clear_queue()
            
            # Add queries to queue
            for query in test_queries:
                self.prioritizer.add_to_priority_queue(query)
            
            # Extract queries and verify order
            extracted = []
            while not self.prioritizer.get_queue_status()['is_empty']:
                query = self.prioritizer.get_next_query()
                if query:
                    extracted.append(query)
            
            # Check that critical/high priority queries come first
            first_query_priority = self.prioritizer.get_priority(extracted[0])
            if first_query_priority in [Priority.CRITICAL, Priority.HIGH]:
                results['passed'] += 1
                self.log("✓ Priority queue ordering works correctly")
            else:
                results['failed'] += 1
                error_msg = f"First query priority should be CRITICAL/HIGH, got {first_query_priority.name}"
                results['errors'].append(f"Priority Queue: {error_msg}")
                self.log(f"✗ Priority Queue: {error_msg}", "ERROR")
                
        except Exception as e:
            results['failed'] += 1
            error_msg = f"Priority queue exception: {str(e)}"
            results['errors'].append(f"Priority Queue: {error_msg}")
            self.log(f"✗ Priority Queue: {error_msg}", "ERROR")
        
        return results
    
    def run_statistics_tests(self) -> Dict[str, Any]:
        """Test statistics and analytics functionality."""
        self.log("Running statistics tests...")
        
        results = {
            'name': 'Statistics and Analytics',
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Clear any existing stats
        self.prioritizer.reset_stats()
        
        # Record some test statistics
        test_data = [
            ("What is AI?", 0.3, True),
            ("Emergency server down!", 1.2, False),
            ("How to implement OAuth?", 0.6, True),
            ("Hello there", 0.1, True),
            ("Critical database error", 0.8, False)
        ]
        
        try:
            for query, time_taken, success in test_data:
                self.prioritizer.record_query_stats(query, time_taken, success)
            
            insights = self.prioritizer.get_optimization_insights()
            
            # Check basic metrics
            if insights['total_queries'] == len(test_data):
                results['passed'] += 1
                self.log("✓ Query statistics recording works")
            else:
                results['failed'] += 1
                error_msg = f"Expected {len(test_data)} queries, got {insights['total_queries']}"
                results['errors'].append(f"Stats Recording: {error_msg}")
                self.log(f"✗ Stats Recording: {error_msg}", "ERROR")
            
            # Check success rate calculation
            expected_success_rate = 3/5  # 3 successes out of 5
            actual_success_rate = insights['overall_success_rate']
            if abs(actual_success_rate - expected_success_rate) < 0.01:
                results['passed'] += 1
                self.log("✓ Success rate calculation works")
            else:
                results['failed'] += 1
                error_msg = f"Expected success rate {expected_success_rate}, got {actual_success_rate}"
                results['errors'].append(f"Success Rate: {error_msg}")
                self.log(f"✗ Success Rate: {error_msg}", "ERROR")
            
            # Check that insights structure is complete
            required_keys = ['total_queries', 'avg_processing_time', 'overall_success_rate', 
                           'slowest_query_types', 'most_common_query_types', 'recommendations']
            
            missing_keys = [key for key in required_keys if key not in insights]
            if not missing_keys:
                results['passed'] += 1
                self.log("✓ Insights structure is complete")
            else:
                results['failed'] += 1
                error_msg = f"Missing keys in insights: {missing_keys}"
                results['errors'].append(f"Insights Structure: {error_msg}")
                self.log(f"✗ Insights Structure: {error_msg}", "ERROR")
                
        except Exception as e:
            results['failed'] += 1
            error_msg = f"Statistics exception: {str(e)}"
            results['errors'].append(f"Statistics: {error_msg}")
            self.log(f"✗ Statistics: {error_msg}", "ERROR")
        
        return results
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        self.log("Running performance benchmarks...")
        
        results = {
            'name': 'Performance Benchmarks',
            'benchmarks': {},
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Single query performance
        try:
            test_query = "What is the best approach for implementing machine learning?"
            iterations = 1000
            
            start_time = time.time()
            for _ in range(iterations):
                self.prioritizer.get_priority(test_query)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iterations
            queries_per_second = 1 / avg_time if avg_time > 0 else 0
            
            results['benchmarks']['single_query_avg_time'] = avg_time
            results['benchmarks']['queries_per_second'] = queries_per_second
            
            if avg_time < 0.001:  # Less than 1ms per query
                results['passed'] += 1
                self.log(f"✓ Single query performance: {avg_time*1000:.3f}ms avg, {queries_per_second:.0f} qps")
            else:
                results['failed'] += 1
                error_msg = f"Single query too slow: {avg_time*1000:.3f}ms"
                results['errors'].append(f"Single Query Performance: {error_msg}")
                self.log(f"✗ Single Query Performance: {error_msg}", "ERROR")
                
        except Exception as e:
            results['failed'] += 1
            error_msg = f"Single query benchmark exception: {str(e)}"
            results['errors'].append(f"Single Query Benchmark: {error_msg}")
            self.log(f"✗ Single Query Benchmark: {error_msg}", "ERROR")
        
        # Batch sorting performance
        try:
            test_queries = [f"Query number {i} with varying content" for i in range(1000)]
            
            start_time = time.time()
            sorted_queries = self.prioritizer.sort_queries_by_priority(test_queries)
            end_time = time.time()
            
            sorting_time = end_time - start_time
            queries_sorted_per_second = len(test_queries) / sorting_time if sorting_time > 0 else 0
            
            results['benchmarks']['batch_sorting_time'] = sorting_time
            results['benchmarks']['queries_sorted_per_second'] = queries_sorted_per_second
            
            if sorting_time < 1.0 and len(sorted_queries) == 1000:  # Less than 1 second for 1000 queries
                results['passed'] += 1
                self.log(f"✓ Batch sorting performance: {sorting_time:.3f}s for 1000 queries")
            else:
                results['failed'] += 1
                error_msg = f"Batch sorting too slow: {sorting_time:.3f}s"
                results['errors'].append(f"Batch Sorting Performance: {error_msg}")
                self.log(f"✗ Batch Sorting Performance: {error_msg}", "ERROR")
                
        except Exception as e:
            results['failed'] += 1
            error_msg = f"Batch sorting benchmark exception: {str(e)}"
            results['errors'].append(f"Batch Sorting Benchmark: {error_msg}")
            self.log(f"✗ Batch Sorting Benchmark: {error_msg}", "ERROR")
        
        return results
    
    def run_real_world_scenarios(self) -> Dict[str, Any]:
        """Run real-world scenario tests."""
        self.log("Running real-world scenario tests...")
        
        results = {
            'name': 'Real-World Scenarios',
            'scenarios': {},
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Customer support scenario
        try:
            support_results = self.scenarios.customer_support_scenario(self.prioritizer)
            results['scenarios']['customer_support'] = support_results
            
            # Validate results
            if (support_results['first_priority'] in ['CRITICAL', 'HIGH'] and 
                support_results['total_tickets'] == len(support_results['processing_order'])):
                results['passed'] += 1
                self.log(f"✓ Customer support scenario: {support_results['total_tickets']} tickets processed")
            else:
                results['failed'] += 1
                error_msg = "Customer support scenario validation failed"
                results['errors'].append(f"Customer Support: {error_msg}")
                self.log(f"✗ Customer Support: {error_msg}", "ERROR")
                
        except Exception as e:
            results['failed'] += 1
            error_msg = f"Customer support scenario exception: {str(e)}"
            results['errors'].append(f"Customer Support: {error_msg}")
            self.log(f"✗ Customer Support: {error_msg}", "ERROR")
        
        # Development team scenario  
        try:
            dev_results = self.scenarios.development_team_scenario(self.prioritizer)
            results['scenarios']['development_team'] = dev_results
            
            # Validate results
            if (len(dev_results['queries_by_hour']) == 8 and 
                dev_results['priority_distribution']['CRITICAL'] > 0):
                results['passed'] += 1
                self.log("✓ Development team scenario completed successfully")
            else:
                results['failed'] += 1
                error_msg = "Development team scenario validation failed"
                results['errors'].append(f"Development Team: {error_msg}")
                self.log(f"✗ Development Team: {error_msg}", "ERROR")
                
        except Exception as e:
            results['failed'] += 1
            error_msg = f"Development team scenario exception: {str(e)}"
            results['errors'].append(f"Development Team: {error_msg}")
            self.log(f"✗ Development Team: {error_msg}", "ERROR")
        
        return results
    
    def run_edge_case_tests(self) -> Dict[str, Any]:
        """Test edge cases and error handling."""
        self.log("Running edge case tests...")
        
        results = {
            'name': 'Edge Cases and Error Handling',
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        edge_cases = [
            ("", "Empty string"),
            ("   ", "Whitespace only"),
            ("\n\t\r", "Control characters"),
            ("a" * 1000, "Very long string"),
            ("🚨💻🔥" * 50, "Many emojis"),
            ("SELECT * FROM users;", "SQL-like content"),
            ("<script>alert('test')</script>", "HTML/JS content"),
            ("null\x00byte", "Null byte"),
            ("Здравствуйте!", "Cyrillic text"),
            ("مرحبا", "Arabic text"),
            ("こんにちは", "Japanese text"),
            ("¿Cómo está?", "Spanish with accents"),
        ]
        
        for test_input, description in edge_cases:
            try:
                priority = self.prioritizer.get_priority(test_input)
                if isinstance(priority, Priority):
                    results['passed'] += 1
                    self.log(f"✓ {description}: {priority.name}")
                else:
                    results['failed'] += 1
                    error_msg = f"Invalid priority type returned: {type(priority)}"
                    results['errors'].append(f"{description}: {error_msg}")
                    self.log(f"✗ {description}: {error_msg}", "ERROR")
                    
            except Exception as e:
                results['failed'] += 1
                error_msg = f"Exception: {str(e)}"
                results['errors'].append(f"{description}: {error_msg}")
                self.log(f"✗ {description}: {error_msg}", "ERROR")
        
        return results
    
    def run_load_testing(self, query_count: int = 1000) -> Dict[str, Any]:
        """Run load testing with many queries."""
        self.log(f"Running load testing with {query_count} queries...")
        
        results = {
            'name': f'Load Testing ({query_count} queries)',
            'load_results': {},
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            load_results = self.scenarios.load_testing_scenario(self.prioritizer, query_count)
            results['load_results'] = load_results
            
            # Validate performance under load
            accuracy = load_results['accuracy']['accuracy']
            throughput = load_results['throughput']['queries_per_second']
            
            if accuracy > 0.8:
                results['passed'] += 1
                self.log(f"✓ Load test accuracy: {accuracy:.2%}")
            else:
                results['failed'] += 1
                error_msg = f"Accuracy too low under load: {accuracy:.2%}"
                results['errors'].append(f"Load Test Accuracy: {error_msg}")
                self.log(f"✗ Load Test Accuracy: {error_msg}", "ERROR")
            
            if throughput > 50:  # At least 50 queries per second
                results['passed'] += 1
                self.log(f"✓ Load test throughput: {throughput:.1f} qps")
            else:
                results['failed'] += 1
                error_msg = f"Throughput too low: {throughput:.1f} qps"
                results['errors'].append(f"Load Test Throughput: {error_msg}")
                self.log(f"✗ Load Test Throughput: {error_msg}", "ERROR")
                
        except Exception as e:
            results['failed'] += 1
            error_msg = f"Load testing exception: {str(e)}"
            results['errors'].append(f"Load Testing: {error_msg}")
            self.log(f"✗ Load Testing: {error_msg}", "ERROR")
        
        return results
    
    def run_all_tests(self, include_load_test: bool = True, load_test_size: int = 1000) -> Dict[str, Any]:
        """Run all test suites."""
        self.log("Starting comprehensive test suite...")
        start_time = time.time()
        
        all_results = {
            'test_suites': [],
            'summary': {
                'total_passed': 0,
                'total_failed': 0,
                'total_errors': [],
                'execution_time': 0
            }
        }
        
        # Define test suites
        test_suites = [
            ('Basic Functionality', self.run_basic_functionality_tests),
            ('Pattern Recognition', self.run_pattern_recognition_tests),
            ('Sorting and Queue', self.run_sorting_and_queue_tests),
            ('Statistics', self.run_statistics_tests),
            ('Performance Benchmarks', self.run_performance_benchmarks),
            ('Real-World Scenarios', self.run_real_world_scenarios),
            ('Edge Cases', self.run_edge_case_tests),
        ]
        
        if include_load_test:
            test_suites.append(('Load Testing', lambda: self.run_load_testing(load_test_size)))
        
        # Run each test suite
        for suite_name, test_function in test_suites:
            self.log(f"\n--- Running {suite_name} ---")
            try:
                suite_results = test_function()
                all_results['test_suites'].append(suite_results)
                
                # Update summary
                all_results['summary']['total_passed'] += suite_results['passed']
                all_results['summary']['total_failed'] += suite_results['failed']
                all_results['summary']['total_errors'].extend(suite_results['errors'])
                
                # Log suite results
                self.log(f"{suite_name}: {suite_results['passed']} passed, {suite_results['failed']} failed", "RESULT")
                
            except Exception as e:
                error_msg = f"Test suite {suite_name} crashed: {str(e)}"
                self.log(error_msg, "ERROR")
                all_results['summary']['total_errors'].append(error_msg)
                all_results['summary']['total_failed'] += 1
        
        # Calculate execution time
        all_results['summary']['execution_time'] = time.time() - start_time
        
        return all_results
    
    def print_final_report(self, results: Dict[str, Any]):
        """Print a comprehensive final report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        summary = results['summary']
        total_tests = summary['total_passed'] + summary['total_failed']
        success_rate = summary['total_passed'] / total_tests * 100 if total_tests > 0 else 0
        
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {summary['total_passed']} ({summary['total_passed']/total_tests*100:.1f}%)")
        print(f"  Failed: {summary['total_failed']} ({summary['total_failed']/total_tests*100:.1f}%)")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Execution Time: {summary['execution_time']:.2f} seconds")
        
        print(f"\nDETAILED RESULTS BY TEST SUITE:")
        for suite in results['test_suites']:
            suite_total = suite['passed'] + suite['failed']
            suite_success = suite['passed'] / suite_total * 100 if suite_total > 0 else 0
            print(f"  {suite['name']}: {suite['passed']}/{suite_total} ({suite_success:.1f}%)")
        
        # Performance benchmarks
        for suite in results['test_suites']:
            if 'benchmarks' in suite:
                print(f"\nPERFORMANCE BENCHMARKS:")
                benchmarks = suite['benchmarks']
                if 'single_query_avg_time' in benchmarks:
                    print(f"  Single Query Time: {benchmarks['single_query_avg_time']*1000:.3f}ms")
                if 'queries_per_second' in benchmarks:
                    print(f"  Queries per Second: {benchmarks['queries_per_second']:.0f}")
                if 'batch_sorting_time' in benchmarks:
                    print(f"  Batch Sorting (1000): {benchmarks['batch_sorting_time']:.3f}s")
        
        # Load testing results
        for suite in results['test_suites']:
            if 'load_results' in suite:
                load_results = suite['load_results']
                print(f"\nLOAD TESTING RESULTS:")
                print(f"  Query Count: {load_results['query_count']}")
                print(f"  Accuracy: {load_results['accuracy']['accuracy']:.2%}")
                print(f"  Throughput: {load_results['throughput']['queries_per_second']:.1f} qps")
                print(f"  Total Time: {load_results['total_time']:.2f}s")
        
        # Error summary
        if summary['total_errors']:
            print(f"\nERRORS AND FAILURES:")
            for i, error in enumerate(summary['total_errors'][:10], 1):  # Show first 10 errors
                print(f"  {i}. {error}")
            if len(summary['total_errors']) > 10:
                print(f"  ... and {len(summary['total_errors']) - 10} more errors")
        
        # Final verdict
        print(f"\n" + "="*80)
        if success_rate >= 95:
            print("🎉 EXCELLENT: All systems functioning optimally!")
        elif success_rate >= 85:
            print("✅ GOOD: System functioning well with minor issues")
        elif success_rate >= 70:
            print("⚠️  WARNING: System has significant issues that need attention")
        else:
            print("❌ CRITICAL: System has major problems requiring immediate fixes")
        print("="*80)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description='Enhanced Greedy Priority Algorithm Test Suite')
    parser.add_argument('--benchmark', action='store_true', help='Include performance benchmarks')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', action='store_true', help='Quick test run (smaller load test)')
    parser.add_argument('--load-size', type=int, default=1000, help='Load test size (default: 1000)')
    parser.add_argument('--no-load', action='store_true', help='Skip load testing')
    
    args = parser.parse_args()
    
    # Adjust parameters based on arguments
    if args.quick:
        load_size = 100
    else:
        load_size = args.load_size
    
    include_load = not args.no_load
    
    # Create and run test runner
    runner = TestRunner(verbose=args.verbose)
    
    try:
        print("Enhanced Greedy Priority Algorithm - Comprehensive Test Suite")
        print(f"Test Configuration: Load Size={load_size}, Include Load={include_load}")
        print("-" * 80)
        
        # Run all tests
        results = runner.run_all_tests(
            include_load_test=include_load,
            load_test_size=load_size
        )
        
        # Print final report
        runner.print_final_report(results)
        
        # Exit with appropriate code
        if results['summary']['total_failed'] == 0:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Some tests failed
            
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
