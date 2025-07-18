# === greedy_priority.py ===
"""
Milestone 3: Greedy Priority Algorithm

This module implements a greedy algorithm to prioritize user queries based on urgency,
complexity, or intent. By assigning priority levels, the chatbot can optimize which
queries to process first in high-load or batch-processing scenarios.
"""

import heapq
from typing import List, Dict, Tuple, Any
import time


class GreedyPriority:
    """
    Implements greedy query prioritization for optimizing response time and resource usage.
    Priority is determined based on specific keywords or structural features of the query.
    It also tracks statistics for different types of queries to guide optimization.
    """
    
    def __init__(self):
        """
        Initialize the GreedyPriority manager with default priority categories and
        supporting structures for tracking performance.
        """
        # Priority keyword dictionary:
        # Lower number = higher priority.
        self.priority_keywords = {
            1: ['urgent', 'emergency', 'help', 'error', 'problem', 'issue', 'critical', 'important'],  # High
            2: ['question', 'how', 'what', 'explain', 'tell', 'describe', 'define'],                   # Medium
            3: ['hello', 'hi', 'thanks', 'thank you', 'goodbye', 'bye', 'chat', 'talk']               # Low
        }

        # Dictionary to track statistics of queries handled
        self.query_stats = {}

        # Optional heap-based queue for future use in prioritized execution
        self.priority_queue = []
    
    def get_priority(self, query: str) -> int:
        """
        Determine the priority level of a query using a greedy strategy.

        Priority is based on:
            - Keywords
            - Query length
            - Presence of question marks

        Args:
            query (str): The user query to evaluate.

        Returns:
            int: A priority level (1 = high, 3 = low).
        """
        try:
            query_lower = query.lower()

            # Step 1: Check for matching priority keywords
            for priority_level, keywords in self.priority_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    return priority_level
            
            # Step 2: Fallback heuristics
            if len(query) > 100:
                return 1  # Long queries are presumed more complex
            elif len(query) > 50:
                return 2

            # Step 3: If question mark exists, treat as medium priority
            if '?' in query:
                return 2
            
            # Default to medium if no conditions match
            return 2

        except Exception as e:
            print(f"Error in get_priority: {e}")
            return 2
    
    def sort_queries_by_priority(self, queries: List[str]) -> List[Tuple[int, str]]:
        """
        Sort a list of queries based on their priority.

        Args:
            queries (List[str]): A list of user queries.

        Returns:
            List[Tuple[int, str]]: Sorted list of (priority, query) tuples,
                                   from highest to lowest priority.
        """
        try:
            prioritized_queries = []
            for query in queries:
                priority = self.get_priority(query)
                prioritized_queries.append((priority, query))

            # Sort in ascending order of priority number (1 comes before 3)
            prioritized_queries.sort(key=lambda x: x[0])

            return prioritized_queries

        except Exception as e:
            print(f"Error sorting queries: {e}")
            return [(2, query) for query in queries]  # Default all to medium priority if failed

    def record_query_stats(self, query: str, processing_time: float, success: bool) -> None:
        """
        Store statistics for a query after it is processed. This includes
        success rate, average response time, and number of times encountered.

        Args:
            query (str): The original user query.
            processing_time (float): Time taken to process the query.
            success (bool): Whether the response was successful.
        """
        try:
            query_type = self._categorize_query(query)

            # Initialize tracking for this type if it doesn't exist
            if query_type not in self.query_stats:
                self.query_stats[query_type] = {
                    'total_time': 0.0,
                    'count': 0,
                    'success_count': 0,
                    'avg_time': 0.0,
                    'success_rate': 0.0
                }

            stats = self.query_stats[query_type]

            # Update totals
            stats['total_time'] += processing_time
            stats['count'] += 1
            if success:
                stats['success_count'] += 1

            # Update computed metrics
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['success_rate'] = stats['success_count'] / stats['count']

        except Exception as e:
            print(f"Error recording query stats: {e}")
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """
        Generate insights based on collected statistics.

        Returns:
            Dict[str, Any]: Includes:
                - Total query volume
                - Overall average processing time
                - Success rate
                - Slowest query types
                - Most common query types
        """
        try:
            insights = {
                'total_queries': sum(stats['count'] for stats in self.query_stats.values()),
                'avg_processing_time': 0.0,
                'overall_success_rate': 0.0,
                'slowest_query_types': [],
                'most_common_query_types': []
            }

            if not self.query_stats:
                return insights

            total_time = sum(stats['total_time'] for stats in self.query_stats.values())
            total_count = sum(stats['count'] for stats in self.query_stats.values())
            total_success = sum(stats['success_count'] for stats in self.query_stats.values())

            if total_count > 0:
                insights['avg_processing_time'] = total_time / total_count
                insights['overall_success_rate'] = total_success / total_count

            # Top 3 slowest query types
            sorted_by_time = sorted(self.query_stats.items(),
                                    key=lambda x: x[1]['avg_time'],
                                    reverse=True)
            insights['slowest_query_types'] = [
                (qtype, stats['avg_time']) for qtype, stats in sorted_by_time[:3]
            ]

            # Top 3 most frequent query types
            sorted_by_count = sorted(self.query_stats.items(),
                                     key=lambda x: x[1]['count'],
                                     reverse=True)
            insights['most_common_query_types'] = [
                (qtype, stats['count']) for qtype, stats in sorted_by_count[:3]
            ]

            return insights

        except Exception as e:
            print(f"Error getting optimization insights: {e}")
            return {'error': 'Could not generate insights'}
    
    def _categorize_query(self, query: str) -> str:
        """
        Categorize a query into a logical type (e.g., definition, location, how-to) for analytics.

        Args:
            query (str): The user query string

        Returns:
            str: A query type category label
        """
        try:
            query_lower = query.lower()

            if any(word in query_lower for word in ['what', 'define', 'explain']):
                return 'definition'
            elif any(word in query_lower for word in ['how', 'process', 'work']):
                return 'how-to'
            elif any(word in query_lower for word in ['where', 'location', 'place']):
                return 'location'
            elif any(word in query_lower for word in ['when', 'time', 'date']):
                return 'temporal'
            elif any(word in query_lower for word in ['why', 'reason', 'cause']):
                return 'causal'
            elif any(word in query_lower for word in ['who', 'person', 'people']):
                return 'person'
            else:
                return 'general'

        except Exception as e:
            print(f"Error categorizing query: {e}")
            return 'general'
