"""
Test Utilities and Advanced Testing Scenarios for Greedy Priority Algorithm

This module provides utilities for testing the GreedyPriority system and includes
advanced testing scenarios for real-world usage patterns.
"""

import random
import time
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import pytest
from milestones.greedy_priority import GreedyPriority, Priority


@dataclass
class TestQuery:
    """Represents a test query with expected results."""
    text: str
    expected_priority: Priority
    category: str
    description: str


class QueryGenerator:
    """Generates realistic test queries for various scenarios."""
    
    def __init__(self):
        self.query_templates = {
            Priority.CRITICAL: [
                "Emergency: {system} is {failure_type}",
                "Critical error in {component}: {error_desc}",
                "Urgent: {service} {critical_action} immediately",
                "{system} crashed and {consequence}",
                "Data loss detected in {location}",
                "Security breach: {threat_type} in {system}",
                "System down: {error_code} - {impact}"
            ],
            Priority.HIGH: [
                "Need help with {task} by {deadline}",
                "Important: {issue} affecting {scope}",
                "How to fix {problem} quickly?",
                "Priority task: {action} for {project}",
                "Deadline approaching for {deliverable}",
                "Performance issue with {component}",
                "Integration problem between {system1} and {system2}"
            ],
            Priority.MEDIUM: [
                "What is {concept}?",
                "How does {technology} work?",
                "Explain {topic} in detail",
                "Can you help me understand {subject}?",
                "What's the difference between {option1} and {option2}?",
                "Show me how to {action}",
                "Why is {phenomenon} happening?"
            ],
            Priority.LOW: [
                "{greeting}, how are you?",
                "Thanks for {help_type}",
                "{farewell}, have a great day",
                "Just wanted to say {appreciation}",
                "Good {time_of_day}!",
                "Hope you're doing well",
                "Nice to chat with you"
            ]
        }
        
        self.variables = {
            'system': ['database', 'server', 'application', 'network', 'API', 'website'],
            'failure_type': ['down', 'unresponsive', 'corrupted', 'inaccessible', 'frozen'],
            'component': ['authentication module', 'payment system', 'user interface', 'backend'],
            'error_desc': ['connection timeout', 'memory overflow', 'access denied', 'invalid response'],
            'service': ['production server', 'main database', 'web service', 'file system'],
            'critical_action': ['needs restart', 'requires immediate attention', 'must be fixed'],
            'consequence': ['users cannot login', 'data is inaccessible', 'operations halted'],
            'location': ['user accounts', 'transaction records', 'configuration files'],
            'threat_type': ['unauthorized access', 'malware detected', 'data breach'],
            'error_code': ['500', '503', '404', 'CONNECTION_FAILED', 'TIMEOUT'],
            'impact': ['affecting all users', 'blocking transactions', 'service unavailable'],
            'task': ['deployment', 'configuration', 'troubleshooting', 'optimization'],
            'deadline': ['end of day', 'tomorrow morning', 'this week', 'by 5 PM'],
            'issue': ['performance degradation', 'compatibility problem', 'security vulnerability'],
            'scope': ['production environment', 'user experience', 'system stability'],
            'problem': ['authentication issue', 'slow response time', 'memory leak'],
            'action': ['implement feature', 'deploy update', 'configure system'],
            'project': ['mobile app', 'web platform', 'microservices', 'data pipeline'],
            'deliverable': ['code review', 'system upgrade', 'security audit'],
            'concept': ['machine learning', 'blockchain', 'cloud computing', 'DevOps'],
            'technology': ['Docker', 'Kubernetes', 'React', 'Python', 'PostgreSQL'],
            'topic': ['REST APIs', 'database design', 'software architecture'],
            'subject': ['design patterns', 'testing strategies', 'performance optimization'],
            'option1': ['SQL', 'MongoDB', 'Redis', 'REST'],
            'option2': ['NoSQL', 'PostgreSQL', 'Memcached', 'GraphQL'],
            'greeting': ['Hello', 'Hi', 'Hey there', 'Good morning'],
            'help_type': ['your assistance', 'the support', 'helping me out'],
            'farewell': ['Goodbye', 'See you later', 'Take care', 'Bye'],
            'appreciation': ['thank you', 'you\'re awesome', 'great job'],
            'time_of_day': ['morning', 'afternoon', 'evening']
        }
    
    def generate_query(self, priority: Priority, count: int = 1) -> List[TestQuery]:
        """Generate realistic queries for a specific priority level."""
        queries = []
        templates = self.query_templates[priority]
        
        for _ in range(count):
            template = random.choice(templates)
            
            # Fill in template variables
            filled_template = template
            for var_name, var_options in self.variables.items():
                if f'{{{var_name}}}' in filled_template:
                    filled_template = filled_template.replace(
                        f'{{{var_name}}}', random.choice(var_options)
                    )
            
            queries.append(TestQuery(
                text=filled_template,
                expected_priority=priority,
                category=f"{priority.name.lower()}_generated",
                description=f"Generated {priority.name.lower()} priority query"
            ))
        
        return queries
    
    def generate_mixed_batch(self, total_count: int = 100) -> List[TestQuery]:
        """Generate a mixed batch of queries with realistic distribution."""
        # Realistic distribution: most queries are medium/low priority
        distribution = {
            Priority.CRITICAL: max(1, total_count // 20),    # 5%
            Priority.HIGH: max(1, total_count // 10),        # 10%
            Priority.MEDIUM: total_count // 2,               # 50%
            Priority.LOW: total_count - (total_count // 20) - (total_count // 10) - (total_count // 2)  # 35%
        }
        
        all_queries = []
        for priority, count in distribution.items():
            all_queries.extend(self.generate_query(priority, count))
        
        # Shuffle to simulate real-world random order
        random.shuffle(all_queries)
        return all_queries


class PerformanceProfiler:
    """Profiles performance of the GreedyPriority system."""
    
    def __init__(self):
        self.results = {}
    
    def profile_priority_calculation(self, prioritizer: GreedyPriority, 
                                   queries: List[str], iterations: int = 1000) -> Dict[str, float]:
        """Profile priority calculation performance."""
        total_time = 0
        for _ in range(iterations):
            start = time.time()
            for query in queries:
                prioritizer.get_priority(query)
            total_time += time.time() - start
        
        avg_time_per_batch = total_time / iterations
        avg_time_per_query = avg_time_per_batch / len(queries)
        
        return {
            'total_time': total_time,
            'avg_time_per_batch': avg_time_per_batch,
            'avg_time_per_query': avg_time_per_query,
            'queries_per_second': 1 / avg_time_per_query if avg_time_per_query > 0 else 0
        }
    
    def profile_sorting(self, prioritizer: GreedyPriority, 
                       queries: List[str], iterations: int = 100) -> Dict[str, float]:
        """Profile query sorting performance."""
        total_time = 0
        for _ in range(iterations):
            start = time.time()
            prioritizer.sort_queries_by_priority(queries)
            total_time += time.time() - start
        
        avg_time = total_time / iterations
        
        return {
            'total_time': total_time,
            'avg_time_per_sort': avg_time,
            'queries_sorted_per_second': len(queries) / avg_time if avg_time > 0 else 0
        }
    
    def profile_queue_operations(self, prioritizer: GreedyPriority, 
                               queries: List[str]) -> Dict[str, float]:
        """Profile priority queue operations."""
        # Profile insertions
        start = time.time()
        for query in queries:
            prioritizer.add_to_priority_queue(query)
        insertion_time = time.time() - start
        
        # Profile extractions
        start = time.time()
        extracted_count = 0
        while not prioritizer.get_queue_status()['is_empty']:
            query = prioritizer.get_next_query()
            if query:
                extracted_count += 1
        extraction_time = time.time() - start
        
        return {
            'insertion_time': insertion_time,
            'extraction_time': extraction_time,
            'total_queue_time': insertion_time + extraction_time,
            'insertions_per_second': len(queries) / insertion_time if insertion_time > 0 else 0,
            'extractions_per_second': extracted_count / extraction_time if extraction_time > 0 else 0
        }


class ValidationHelper:
    """Helper class for validating GreedyPriority behavior."""
    
    @staticmethod
    def validate_priority_consistency(prioritizer: GreedyPriority, 
                                    test_queries: List[TestQuery]) -> Dict[str, Any]:
        """Validate that priority assignments are consistent with expectations."""
        results = {
            'total_tested': len(test_queries),
            'correct': 0,
            'incorrect': 0,
            'mismatches': [],
            'accuracy': 0.0
        }
        
        for test_query in test_queries:
            actual_priority = prioritizer.get_priority(test_query.text)
            
            if actual_priority == test_query.expected_priority:
                results['correct'] += 1
            else:
                results['incorrect'] += 1
                results['mismatches'].append({
                    'query': test_query.text,
                    'expected': test_query.expected_priority.name,
                    'actual': actual_priority.name,
                    'category': test_query.category
                })
        
        results['accuracy'] = results['correct'] / results['total_tested'] if results['total_tested'] > 0 else 0
        return results
    
    @staticmethod
    def validate_sorting_order(sorted_queries: List[Tuple[Priority, str]]) -> bool:
        """Validate that queries are sorted in correct priority order."""
        priorities = [priority for priority, _ in sorted_queries]
        return priorities == sorted(priorities)
    
    @staticmethod
    def validate_queue_order(prioritizer: GreedyPriority, 
                           test_queries: List[TestQuery]) -> Dict[str, Any]:
        """Validate that priority queue processes queries in correct order."""
        # Add all queries to queue
        for test_query in test_queries:
            prioritizer.add_to_priority_queue(test_query.text)
        
        # Extract queries and check order
        extracted_queries = []
        while not prioritizer.get_queue_status()['is_empty']:
            query = prioritizer.get_next_query()
            if query:
                extracted_queries.append(query)
        
        # Validate that higher priority queries come first
        results = {
            'total_processed': len(extracted_queries),
            'correct_order': True,
            'first_critical_position': None,
            'first_high_position': None,
            'order_violations': []
        }
        
        # Find positions of different priority levels
        for i, query in enumerate(extracted_queries):
            # Find the test query that matches this extracted query
            matching_test = next((tq for tq in test_queries if tq.text == query), None)
            if matching_test:
                if matching_test.expected_priority == Priority.CRITICAL and results['first_critical_position'] is None:
                    results['first_critical_position'] = i
                elif matching_test.expected_priority == Priority.HIGH and results['first_high_position'] is None:
                    results['first_high_position'] = i
        
        # Check for order violations
        prev_priority = Priority.CRITICAL
        for i, query in enumerate(extracted_queries):
            matching_test = next((tq for tq in test_queries if tq.text == query), None)
            if matching_test:
                current_priority = matching_test.expected_priority
                if current_priority.value < prev_priority.value:  # Lower number = higher priority
                    results['order_violations'].append({
                        'position': i,
                        'query': query,
                        'priority': current_priority.name,
                        'previous_priority': prev_priority.name
                    })
                    results['correct_order'] = False
                prev_priority = current_priority
        
        return results


class RealWorldScenarios:
    """Test scenarios that simulate real-world usage patterns."""
    
    def __init__(self):
        self.generator = QueryGenerator()
        self.profiler = PerformanceProfiler()
        self.validator = ValidationHelper()
    
    def customer_support_scenario(self, prioritizer: GreedyPriority) -> Dict[str, Any]:
        """Simulate a customer support ticket queue scenario."""
        
        # Generate realistic support tickets
        tickets = [
            TestQuery("My account is locked and I can't access anything!", Priority.HIGH, "support", "Account lockout"),
            TestQuery("Hello, I have a question about billing", Priority.MEDIUM, "support", "Billing inquiry"),
            TestQuery("Thanks for fixing my issue yesterday", Priority.LOW, "support", "Thank you message"),
            TestQuery("URGENT: Payment processing is down for all customers!", Priority.CRITICAL, "support", "System outage"),
            TestQuery("How do I change my password?", Priority.MEDIUM, "support", "Password help"),
            TestQuery("Website is loading very slowly", Priority.HIGH, "support", "Performance issue"),
            TestQuery("Good morning! Hope you're well", Priority.LOW, "support", "Greeting"),
            TestQuery("Critical security breach detected in user data!", Priority.CRITICAL, "support", "Security incident"),
            TestQuery("Can you explain how the new feature works?", Priority.MEDIUM, "support", "Feature question"),
            TestQuery("Bye, have a great day!", Priority.LOW, "support", "Goodbye")
        ]
        
        # Test sorting and processing
        queries_text = [ticket.text for ticket in tickets]
        sorted_queries = prioritizer.sort_queries_by_priority(queries_text)
        
        # Validate that critical issues come first
        first_query_priority = prioritizer.get_priority(sorted_queries[0][1])
        
        # Process through queue
        for ticket in tickets:
            prioritizer.add_to_priority_queue(ticket.text)
        
        processing_order = []
        while not prioritizer.get_queue_status()['is_empty']:
            query = prioritizer.get_next_query()
            if query:
                processing_order.append(query)
                # Simulate processing time based on priority
                priority = prioritizer.get_priority(query)
                if priority == Priority.CRITICAL:
                    processing_time = random.uniform(0.1, 0.3)  # Fast response for critical
                elif priority == Priority.HIGH:
                    processing_time = random.uniform(0.2, 0.5)
                elif priority == Priority.MEDIUM:
                    processing_time = random.uniform(0.3, 0.8)
                else:
                    processing_time = random.uniform(0.1, 0.2)  # Quick for greetings
                
                success = random.choice([True, True, True, False])  # 75% success rate
                prioritizer.record_query_stats(query, processing_time, success)
        
        return {
            'scenario': 'customer_support',
            'total_tickets': len(tickets),
            'first_priority': first_query_priority.name,
            'processing_order': processing_order,
            'insights': prioritizer.get_optimization_insights()
        }
    
    def development_team_scenario(self, prioritizer: GreedyPriority) -> Dict[str, Any]:
        """Simulate a development team's query prioritization."""
        
        dev_queries = [
            TestQuery("Production server crashed - all services down!", Priority.CRITICAL, "dev", "Production outage"),
            TestQuery("How to implement OAuth2 authentication?", Priority.MEDIUM, "dev", "Technical question"),
            TestQuery("Code review needed for PR #123", Priority.HIGH, "dev", "Code review"),
            TestQuery("Thanks for helping with the deployment!", Priority.LOW, "dev", "Appreciation"),
            TestQuery("Memory leak causing server instability", Priority.HIGH, "dev", "Performance issue"),
            TestQuery("What's the difference between Docker and Kubernetes?", Priority.MEDIUM, "dev", "Learning"),
            TestQuery("Database corruption detected in production!", Priority.CRITICAL, "dev", "Data integrity"),
            TestQuery("How's everyone doing today?", Priority.LOW, "dev", "Social"),
            TestQuery("Need help with CI/CD pipeline setup", Priority.HIGH, "dev", "Infrastructure"),
            TestQuery("Explain microservices architecture patterns", Priority.MEDIUM, "dev", "Architecture"),
            TestQuery("Security vulnerability found in dependencies!", Priority.CRITICAL, "dev", "Security"),
            TestQuery("Good job on the release!", Priority.LOW, "dev", "Congratulations")
        ]
        
        # Simulate processing throughout the day
        results = {
            'scenario': 'development_team',
            'queries_by_hour': {},
            'priority_distribution': {p.name: 0 for p in Priority},
            'response_times': {p.name: [] for p in Priority}
        }
        
        # Simulate 8-hour workday
        for hour in range(9, 17):  # 9 AM to 5 PM
            hour_queries = random.sample(dev_queries, random.randint(2, 4))
            results['queries_by_hour'][f'{hour}:00'] = []
            
            for query in hour_queries:
                priority = prioritizer.get_priority(query.text)
                results['priority_distribution'][priority.name] += 1
                
                # Simulate processing
                if priority == Priority.CRITICAL:
                    response_time = random.uniform(0.05, 0.15)  # Immediate response
                elif priority == Priority.HIGH:
                    response_time = random.uniform(0.1, 0.4)
                elif priority == Priority.MEDIUM:
                    response_time = random.uniform(0.3, 1.0)
                else:
                    response_time = random.uniform(0.05, 0.2)
                
                results['response_times'][priority.name].append(response_time)
                results['queries_by_hour'][f'{hour}:00'].append({
                    'query': query.text,
                    'priority': priority.name,
                    'response_time': response_time
                })
                
                success = random.choice([True] * 9 + [False])  # 90% success rate for dev team
                prioritizer.record_query_stats(query.text, response_time, success)
        
        return results
    
    def load_testing_scenario(self, prioritizer: GreedyPriority, 
                            query_count: int = 1000) -> Dict[str, Any]:
        """Simulate high-load conditions with many concurrent queries."""
        
        # Generate large batch of mixed queries
        mixed_queries = self.generator.generate_mixed_batch(query_count)
        query_texts = [q.text for q in mixed_queries]
        
        # Performance profiling
        start_time = time.time()
        
        # Test priority calculation under load
        priority_perf = self.profiler.profile_priority_calculation(
            prioritizer, query_texts[:100], iterations=10
        )
        
        # Test sorting under load
        sorting_perf = self.profiler.profile_sorting(
            prioritizer, query_texts, iterations=5
        )
        
        # Test queue operations under load
        queue_perf = self.profiler.profile_queue_operations(
            prioritizer, query_texts
        )
        
        total_time = time.time() - start_time
        
        # Validate accuracy under load
        accuracy_results = self.validator.validate_priority_consistency(
            prioritizer, mixed_queries
        )
        
        return {
            'scenario': 'load_testing',
            'query_count': query_count,
            'total_time': total_time,
            'performance': {
                'priority_calculation': priority_perf,
                'sorting': sorting_perf,
                'queue_operations': queue_perf
            },
            'accuracy': accuracy_results,
            'throughput': {
                'queries_per_second': query_count / total_time,
                'priorities_per_second': priority_perf['queries_per_second']
            }
        }


# === Advanced Test Cases ===

class TestAdvancedScenarios:
    """Advanced test scenarios for comprehensive validation."""
    
    def setup_method(self):
        """Setup for advanced tests."""
        self.prioritizer = GreedyPriority()
        self.generator = QueryGenerator()
        self.scenarios = RealWorldScenarios()
        self.validator = ValidationHelper()
    
    def test_customer_support_workflow(self):
        """Test complete customer support workflow."""
        results = self.scenarios.customer_support_scenario(self.prioritizer)
        
        # Validate that critical issues are prioritized
        assert results['first_priority'] in ['CRITICAL', 'HIGH'], \
            f"First processed query should be high priority, got {results['first_priority']}"
        
        # Check that all tickets were processed
        assert results['total_tickets'] == len(results['processing_order']), \
            "All tickets should be processed"
        
        # Verify insights are generated
        insights = results['insights']
        assert insights['total_queries'] > 0, "Should have recorded query statistics"
        assert 'recommendations' in insights, "Should provide recommendations"
    
    def test_development_team_workflow(self):
        """Test development team query handling workflow."""
        results = self.scenarios.development_team_scenario(self.prioritizer)
        
        # Validate priority distribution makes sense
        priority_dist = results['priority_distribution']
        assert priority_dist['CRITICAL'] > 0, "Should have some critical queries"
        assert priority_dist['MEDIUM'] > 0, "Should have some medium queries"
        
        # Validate response times are reasonable
        response_times = results['response_times']
        critical_times = response_times['CRITICAL']
        medium_times = response_times['MEDIUM']
        
        if critical_times and medium_times:
            avg_critical = sum(critical_times) / len(critical_times)
            avg_medium = sum(medium_times) / len(medium_times)
            assert avg_critical <= avg_medium, "Critical queries should have faster response times"
        
        # Check hourly distribution
        assert len(results['queries_by_hour']) == 8, "Should have 8 hours of data"
    
    def test_load_testing_performance(self):
        """Test system performance under load."""
        results = self.scenarios.load_testing_scenario(self.prioritizer, query_count=500)
        
        # Validate performance metrics
        perf = results['performance']
        
        # Priority calculation should be fast
        assert perf['priority_calculation']['queries_per_second'] > 1000, \
            f"Priority calculation too slow: {perf['priority_calculation']['queries_per_second']} qps"
        
        # Sorting should handle reasonable loads
        assert perf['sorting']['queries_sorted_per_second'] > 100, \
            f"Sorting too slow: {perf['sorting']['queries_sorted_per_second']} qps"
        
        # Queue operations should be efficient
        assert perf['queue_operations']['insertions_per_second'] > 1000, \
            f"Queue insertions too slow: {perf['queue_operations']['insertions_per_second']} ips"
        
        # Accuracy should remain high under load
        accuracy = results['accuracy']['accuracy']
        assert accuracy > 0.8, f"Accuracy too low under load: {accuracy:.2%}"
        
        # Overall throughput should be reasonable
        assert results['throughput']['queries_per_second'] > 50, \
            f"Overall throughput too low: {results['throughput']['queries_per_second']} qps"
    
    def test_priority_consistency_across_variations(self):
        """Test that priority assignment is consistent across query variations."""
        base_queries = [
            "Help me with this problem",
            "Emergency server issue",
            "What is machine learning?",
            "Thanks for your help"
        ]
        
        # Generate variations of each query
        variations = {
            "Help me with this problem": [
                "Help me with this problem please",
                "Can you help me with this problem?",
                "I need help with this problem",
                "Help me with this problem ASAP"
            ],
            "Emergency server issue": [
                "EMERGENCY: Server issue!",
                "Emergency - server issue detected",
                "Server issue - this is an emergency",
                "Emergency situation: server issue"
            ],
            "What is machine learning?": [
                "What is machine learning",
                "Could you explain what machine learning is?",
                "Tell me about machine learning",
                "Machine learning - what is it?"
            ],
            "Thanks for your help": [
                "Thank you for your help",
                "Thanks for helping me",
                "I appreciate your help",
                "Thanks so much for the help!"
            ]
        }
        
        # Test consistency within each group
        for base_query, query_variations in variations.items():
            base_priority = self.prioritizer.get_priority(base_query)
            
            for variation in query_variations:
                variation_priority = self.prioritizer.get_priority(variation)
                
                # Allow some flexibility (within 1 priority level)
                priority_diff = abs(base_priority.value - variation_priority.value)
                assert priority_diff <= 1, \
                    f"Priority inconsistency: '{base_query}' ({base_priority.name}) vs " \
                    f"'{variation}' ({variation_priority.name})"
    
    def test_multilingual_query_handling(self):
        """Test handling of queries in different languages and with mixed content."""
        multilingual_queries = [
            "¡Emergencia! El servidor está caído",  # Spanish emergency
            "Помогите, пожалуйста, с этой проблемой",  # Russian help request  
            "什么是人工智能？",  # Chinese question
            "Merci beaucoup pour votre aide",  # French thanks
            "Hello, can you help with this 紧急 problem?",  # Mixed English-Chinese
            "🚨 URGENT: System down! 🚨",  # With emojis
            "ERROR 500: Database connection failed",  # Technical with code
        ]
        
        for query in multilingual_queries:
            try:
                priority = self.prioritizer.get_priority(query)
                assert isinstance(priority, Priority), \
                    f"Should handle multilingual query: {query}"
                
                # Should be able to sort these queries
                sorted_result = self.prioritizer.sort_queries_by_priority([query])
                assert len(sorted_result) == 1, \
                    f"Should sort multilingual query: {query}"
                
            except Exception as e:
                pytest.fail(f"Failed to handle multilingual query '{query}': {e}")
    
    def test_edge_case_resilience(self):
        """Test system resilience with various edge cases."""
        edge_cases = [
            "",  # Empty string
            " " * 1000,  # Very long whitespace
            "a" * 10000,  # Extremely long query
            "\n\t\r\n\t",  # Only whitespace characters
            "SELECT * FROM users; DROP TABLE users;",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "../../etc/passwd",  # Path traversal attempt
            "null\x00byte",  # Null byte
            "emoji🚀test🎉string🔥",  # Mixed emoji content
            "ÄÖÜÁÉÍÓÚÑ",  # Accented characters
            "الأمان والحماية",  # Arabic text
            "प्राथमिकता",  # Hindi text
            "приоритет",  # Cyrillic text
            "🚨" * 100,  # Many emojis
            "?" * 50,  # Many question marks
            "!" * 50,  # Many exclamation marks
        ]
        
        for edge_case in edge_cases:
            try:
                # Should not crash on any input
                priority = self.prioritizer.get_priority(edge_case)
                assert isinstance(priority, Priority), \
                    f"Should return valid priority for edge case: {repr(edge_case)}"
                
                # Should handle in batch operations
                batch_result = self.prioritizer.sort_queries_by_priority([edge_case])
                assert len(batch_result) <= 1, \
                    f"Batch processing should handle edge case: {repr(edge_case)}"
                
                # Should handle in queue operations
                self.prioritizer.add_to_priority_queue(edge_case)
                retrieved = self.prioritizer.get_next_query()
                # Retrieved might be None for empty strings, which is acceptable
                
            except Exception as e:
                pytest.fail(f"System should be resilient to edge case '{repr(edge_case)}': {e}")
    
    def test_statistical_accuracy_validation(self):
        """Test statistical accuracy of priority assignments."""
        # Generate large sample of test queries
        test_queries = []
        for priority in Priority:
            test_queries.extend(self.generator.generate_query(priority, 100))
        
        # Validate accuracy
        results = self.validator.validate_priority_consistency(self.prioritizer, test_queries)
        
        # Should achieve high accuracy
        assert results['accuracy'] > 0.85, \
            f"Statistical accuracy too low: {results['accuracy']:.2%}"
        
        # Analyze mismatches by category
        mismatches_by_expected = {}
        for mismatch in results['mismatches']:
            expected = mismatch['expected']
            if expected not in mismatches_by_expected:
                mismatches_by_expected[expected] = 0
            mismatches_by_expected[expected] += 1
        
        # No priority level should have more than 20% mismatches
        for priority in Priority:
            priority_name = priority.name
            priority_queries = [tq for tq in test_queries if tq.expected_priority == priority]
            mismatches = mismatches_by_expected.get(priority_name, 0)
            
            if priority_queries:
                mismatch_rate = mismatches / len(priority_queries)
                assert mismatch_rate < 0.2, \
                    f"Too many mismatches for {priority_name}: {mismatch_rate:.2%}"
    
    def test_performance_regression_monitoring(self):
        """Test for performance regression monitoring."""
        # Baseline performance test
        baseline_queries = ["What is AI?"] * 1000
        
        start_time = time.time()
        for query in baseline_queries:
            self.prioritizer.get_priority(query)
        baseline_time = time.time() - start_time
        
        # Performance should be consistent
        start_time = time.time()
        for query in baseline_queries:
            self.prioritizer.get_priority(query)
        second_run_time = time.time() - start_time
        
        # Performance should not degrade significantly (allow 20% variance)
        performance_ratio = second_run_time / baseline_time
        assert 0.8 <= performance_ratio <= 1.2, \
            f"Performance regression detected: {performance_ratio:.2f}x"
        
        # Memory usage should be reasonable (test indirectly through behavior)
        large_batch = self.generator.generate_mixed_batch(1000)
        query_texts = [q.text for q in large_batch]
        
        # Should handle large batches without issues
        start_time = time.time()
        sorted_result = self.prioritizer.sort_queries_by_priority(query_texts)
        processing_time = time.time() - start_time
        
        assert len(sorted_result) == 1000, "Should process all queries in large batch"
        assert processing_time < 5.0, f"Large batch processing too slow: {processing_time:.2f}s"


if __name__ == "__main__":
    # Example usage of test utilities
    generator = QueryGenerator()
    prioritizer = GreedyPriority()
    scenarios = RealWorldScenarios()
    
    # Generate and test some queries
    test_queries = generator.generate_mixed_batch(20)
    for query in test_queries[:5]:
        priority = prioritizer.get_priority(query.text)
        print(f"Query: {query.text}")
        print(f"Expected: {query.expected_priority.name}, Actual: {priority.name}")
        print(f"Match: {priority == query.expected_priority}")
        print("-" * 50)
    
    # Run a scenario
    print("\nCustomer Support Scenario:")
    support_results = scenarios.customer_support_scenario(prioritizer)
    print(f"Processed {support_results['total_tickets']} tickets")
    print(f"First priority: {support_results['first_priority']}")
    
    print("\nLoad Testing Scenario:")
    load_results = scenarios.load_testing_scenario(prioritizer, 100)
    print(f"Accuracy: {load_results['accuracy']['accuracy']:.2%}")
    print(f"Throughput: {load_results['throughput']['queries_per_second']:.1f} qps")