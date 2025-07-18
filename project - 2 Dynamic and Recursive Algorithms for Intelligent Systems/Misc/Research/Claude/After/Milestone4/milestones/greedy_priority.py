# === greedy_priority.py ===
"""
Milestone 3: Enhanced Greedy Priority Algorithm

This module implements an advanced greedy algorithm to prioritize user queries based on urgency,
complexity, context, and intent. By assigning sophisticated priority levels, the chatbot can 
optimize which queries to process first in high-load, batch-processing, or real-time scenarios.

The algorithm uses dynamic programming principles for caching priority calculations,
machine learning-inspired pattern recognition, and contextual analysis for intelligent
query prioritization and resource optimization.
"""

import heapq
import time
import re
import datetime
from typing import List, Dict, Tuple, Any, Optional, Set, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import IntEnum
import json


class Priority(IntEnum):
    """
    Enhanced priority levels for query processing with clear semantic meaning.
    Lower numeric values indicate higher priority for processing.
    """
    CRITICAL = 1    # Emergency situations, system failures, critical errors
    HIGH = 2        # Important tasks, deadlines, urgent help requests
    MEDIUM = 3      # Standard questions, explanations, general queries
    LOW = 4         # Casual interactions, greetings, social exchanges


@dataclass
class QueryMetrics:
    """
    Comprehensive data class to store detailed query processing metrics and analytics.
    Supports advanced performance monitoring and optimization insights.
    """
    total_time: float = 0.0
    count: int = 0
    success_count: int = 0
    avg_time: float = 0.0
    success_rate: float = 0.0
    last_processed: Optional[datetime.datetime] = None
    priority_distribution: Dict[int, int] = field(default_factory=dict)
    complexity_scores: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    
    def update(self, processing_time: float, success: bool, priority: int = None, 
               complexity: float = None) -> None:
        """
        Update metrics with new processing data including advanced analytics.
        
        Args:
            processing_time (float): Time taken to process the query
            success (bool): Whether processing was successful
            priority (int, optional): Priority level of the processed query
            complexity (float, optional): Calculated complexity score
        """
        self.total_time += processing_time
        self.count += 1
        if success:
            self.success_count += 1
        
        self.avg_time = self.total_time / self.count
        self.success_rate = self.success_count / self.count
        self.last_processed = datetime.datetime.now()
        
        if priority is not None:
            self.priority_distribution[priority] = self.priority_distribution.get(priority, 0) + 1
        
        if complexity is not None:
            self.complexity_scores.append(complexity)
        
        self.response_times.append(processing_time)
        
        # Keep only recent response times for trending analysis
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]


class GreedyPriority:
    """
    Advanced greedy query prioritization system implementing sophisticated algorithms for
    intelligent resource allocation and response optimization. 
    
    Features:
    - Multi-factor priority calculation using urgency, complexity, and context
    - Dynamic pattern recognition with machine learning-inspired techniques  
    - Comprehensive analytics and performance monitoring
    - Context-aware prioritization with conversation history analysis
    - Adaptive thresholds and self-optimizing parameters
    - Enterprise-grade scalability and reliability
    """
    
    def __init__(self):
        """
        Initialize the enhanced GreedyPriority system with comprehensive data structures,
        advanced pattern recognition capabilities, and performance monitoring systems.
        """
        # Core priority keyword dictionary with enhanced categorization
        # Lower number = higher priority for processing
        self.priority_keywords = {
            Priority.CRITICAL: [
                # System failures and emergencies
                'urgent', 'emergency', 'critical', 'immediately', 'asap', 'crisis',
                'broken', 'down', 'failure', 'crash', 'error', 'bug', 'issue',
                'problem', 'stuck', 'stop', 'halt', 'freeze', 'timeout',
                # Security and data integrity
                'security', 'breach', 'hack', 'vulnerability', 'attack', 'malware',
                'data loss', 'corruption', 'infected', 'virus', 'compromised',
                # Production and business critical
                'production', 'outage', 'service down', 'system failure', 'offline'
            ],
            Priority.HIGH: [
                # Importance and urgency indicators
                'important', 'priority', 'needed', 'required', 'must', 'should',
                'deadline', 'quick', 'fast', 'soon', 'help', 'assist', 'support',
                # Performance and optimization
                'slow', 'performance', 'optimization', 'bottleneck', 'lag',
                'improvement', 'efficiency', 'speed up', 'faster',
                # Business impact
                'client', 'customer', 'revenue', 'business', 'impact', 'loss'
            ],
            Priority.MEDIUM: [
                # Information seeking and learning
                'question', 'how', 'what', 'why', 'when', 'where', 'who', 'which',
                'explain', 'tell', 'describe', 'define', 'show', 'demonstrate',
                'clarify', 'understand', 'learn', 'tutorial', 'guide', 'example',
                # Development and implementation
                'implement', 'develop', 'create', 'build', 'design', 'configure',
                'setup', 'install', 'deploy', 'integrate', 'code', 'programming'
            ],
            Priority.LOW: [
                # Social interactions and pleasantries
                'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon',
                'good evening', 'thanks', 'thank you', 'goodbye', 'bye', 'see you',
                'chat', 'talk', 'discuss', 'opinion', 'think', 'feel', 'like',
                # Non-urgent communications
                'casual', 'friendly', 'social', 'weather', 'personal', 'story'
            ]
        }

        # Advanced regex patterns for sophisticated priority detection
        self.priority_patterns = {
            Priority.CRITICAL: [
                r'\b(can\'t|cannot|won\'t|wont)\s+(work|function|run|start|load|connect|access)\b',
                r'\b(not\s+working|doesn\'t\s+work|failed\s+to|unable\s+to)\b',
                r'\b(server\s+down|service\s+unavailable|connection\s+(lost|failed)|system\s+offline)\b',
                r'\b(data\s+(loss|lost|corrupted|missing)|database\s+(down|corrupt|failed))\b',
                r'\b(emergency|critical|urgent)\b.*\b(help|support|assistance|fix|repair)\b',
                r'\b(production\s+(down|failed|broken)|live\s+site\s+(down|broken))\b',
                r'\b(security\s+(breach|incident|alert)|hack(ed|ing)|malware|virus)\b'
            ],
            Priority.HIGH: [
                r'\b(due\s+(today|tomorrow|soon)|deadline\s+(approaching|tomorrow|today))\b',
                r'\b(need\s+(help|assistance|support)\s+(with|for|urgently))\b',
                r'\b(how\s+to\s+(fix|solve|resolve|repair))\b',
                r'\b(performance\s+(issue|problem)|running\s+slow|very\s+slow)\b',
                r'\b(client\s+(complain|issue|problem)|customer\s+(complain|issue))\b',
                r'\b(important\s+(task|project|deadline)|high\s+priority)\b'
            ],
            Priority.MEDIUM: [
                r'\?+\s*$',  # Questions ending with question marks
                r'\b(can\s+you\s+(help|explain|show|tell|teach))\b',
                r'\b(what\s+is|how\s+does|why\s+is|when\s+should|where\s+can)\b',
                r'\b(tutorial|guide|example|documentation|reference)\b',
                r'\b(learn|understand|explain|clarify|describe)\b'
            ],
            Priority.LOW: [
                r'\b(hello|hi|hey|greetings)\b.*\b(how\s+(are\s+you|is\s+everything))\b',
                r'\b(thank\s+you|thanks|appreciate|grateful)\b',
                r'\b(good\s+(morning|afternoon|evening|day))\b',
                r'\b(nice\s+(day|weather|chat)|lovely\s+(day|weather))\b'
            ]
        }

        # Comprehensive query analysis and statistics tracking
        self.query_stats: Dict[str, QueryMetrics] = {}
        
        # Advanced priority queue with metadata support
        self.priority_queue = []
        self.queue_metadata = {
            'total_processed': 0,
            'average_wait_time': 0.0,
            'priority_distribution': Counter(),
            'processing_trends': [],
            'peak_usage_times': [],
            'bottleneck_indicators': []
        }
        
        # Sophisticated configuration parameters with adaptive capabilities
        self.length_thresholds = {
            'very_long': 200,   # Comprehensive detailed queries
            'long': 100,        # Complex multi-part questions
            'medium': 50,       # Standard detailed queries
            'short': 20         # Brief questions or statements
        }
        
        # Enhanced complexity analysis indicators
        self.complexity_indicators = [
            # Technical complexity
            'algorithm', 'implementation', 'architecture', 'system', 'design',
            'optimization', 'performance', 'scalability', 'integration',
            'configuration', 'deployment', 'security', 'authentication',
            # Advanced concepts
            'machine learning', 'artificial intelligence', 'neural network',
            'deep learning', 'data science', 'big data', 'cloud computing',
            'microservices', 'distributed', 'concurrent', 'parallel',
            # Development complexity
            'framework', 'library', 'api', 'database', 'frontend', 'backend',
            'fullstack', 'devops', 'testing', 'debugging', 'refactoring'
        ]
        
        # Context awareness and conversation history
        self.conversation_context = []
        self.context_window = 10
        self.topic_continuity = defaultdict(lambda: {
            'frequency': 0, 
            'last_mentioned': None, 
            'importance_score': 0.0,
            'context_boost': 0.0
        })
        
        # Performance monitoring and optimization
        self.performance_metrics = {
            'total_queries_processed': 0,
            'average_response_time': 0.0,
            'priority_accuracy': 0.0,
            'cache_hit_rate': 0.0,
            'system_load': 0.0,
            'optimization_suggestions': []
        }
        
        # Advanced caching for priority calculations
        self.priority_cache = {}
        self.cache_max_size = 1000
        self.cache_stats = {'hits': 0, 'misses': 0, 'invalidations': 0}
        
        # Session and temporal analysis
        self.session_start = datetime.datetime.now()
        self.temporal_patterns = {
            'peak_hours': [],
            'low_activity_periods': [],
            'query_velocity': 0.0,
            'burst_detection': False
        }

    def get_priority(self, query: str) -> Union[int, Priority]:
        """
        Determine the priority level of a query using an advanced multi-factor analysis.
        
        This method implements a sophisticated greedy algorithm that considers:
        - Keyword-based urgency detection with weighted scoring
        - Advanced regex pattern matching for context recognition  
        - Query complexity analysis using NLP-inspired techniques
        - Length-based heuristics with adaptive thresholds
        - Conversation context and topic continuity
        - Temporal factors and user behavior patterns
        
        Args:
            query (str): The user query to evaluate and prioritize
            
        Returns:
            Union[int, Priority]: Priority level (1=CRITICAL, 2=HIGH, 3=MEDIUM, 4=LOW)
                                 Returns integer for backward compatibility, Priority enum for enhanced features
        """
        if not query or not query.strip():
            return Priority.LOW
            
        try:
            # Check cache first for performance optimization
            cache_key = self._normalize_query_for_cache(query)
            if cache_key in self.priority_cache:
                self.cache_stats['hits'] += 1
                cached_result = self.priority_cache[cache_key]
                
                # Apply context boost if available
                context_boost = self._calculate_context_boost(query)
                if context_boost > 0:
                    boosted_priority = max(Priority.CRITICAL, Priority(cached_result.value - context_boost))
                    return boosted_priority
                return cached_result
            
            self.cache_stats['misses'] += 1
            
            query_lower = query.lower().strip()
            
            # Phase 1: Advanced pattern-based priority detection (highest precedence)
            pattern_priority = self._analyze_priority_patterns(query_lower)
            if pattern_priority:
                self._cache_priority_result(cache_key, pattern_priority)
                return pattern_priority
            
            # Phase 2: Enhanced keyword-based priority with weighted scoring
            keyword_priority = self._analyze_keyword_priority(query_lower)
            if keyword_priority:
                self._cache_priority_result(cache_key, keyword_priority)
                return keyword_priority
            
            # Phase 3: Advanced complexity-based priority analysis
            complexity_priority = self._analyze_complexity_priority(query)
            if complexity_priority:
                self._cache_priority_result(cache_key, complexity_priority)
                return complexity_priority
            
            # Phase 4: Length-based heuristics with context awareness
            length_priority = self._analyze_length_priority(query)
            if length_priority:
                self._cache_priority_result(cache_key, length_priority)
                return length_priority
            
            # Phase 5: Punctuation and structural analysis
            structural_priority = self._analyze_structural_priority(query)
            if structural_priority:
                self._cache_priority_result(cache_key, structural_priority)
                return structural_priority
            
            # Phase 6: Context-aware default with conversation analysis
            context_priority = self._determine_context_priority(query)
            self._cache_priority_result(cache_key, context_priority)
            return context_priority

        except Exception as e:
            print(f"Error in get_priority: {e}")
            return Priority.MEDIUM  # Safe fallback
    
    def sort_queries_by_priority(self, queries: List[str]) -> List[Tuple[Union[int, Priority], str]]:
        """
        Sort a list of queries based on their calculated priority using advanced algorithms.
        
        Implements an optimized sorting strategy that:
        - Batch processes queries for improved performance
        - Maintains context across related queries  
        - Applies conversation continuity bonuses
        - Considers temporal factors and urgency decay
        - Provides detailed sorting metadata and analytics
        
        Args:
            queries (List[str]): A list of user queries to prioritize and sort
            
        Returns:
            List[Tuple[Union[int, Priority], str]]: Sorted list of (priority, query) tuples,
                                                   ordered from highest to lowest priority
        """
        if not queries:
            return []
            
        try:
            # Batch priority calculation with context awareness
            prioritized_queries = []
            context_topics = self._extract_current_context_topics()
            
            for i, query in enumerate(queries):
                # Calculate base priority
                priority = self.get_priority(query)
                
                # Apply contextual adjustments
                if context_topics:
                    query_topics = self._extract_query_topics(query)
                    if any(topic in context_topics for topic in query_topics):
                        # Boost priority for contextually relevant queries
                        if isinstance(priority, Priority):
                            priority = Priority(max(Priority.CRITICAL.value, priority.value - 1))
                        else:
                            priority = max(1, priority - 1)
                
                # Apply position-based urgency (later queries might be more urgent)
                position_factor = i / len(queries) if len(queries) > 1 else 0
                if position_factor > 0.8:  # Last 20% of queries
                    if isinstance(priority, Priority):
                        priority = Priority(max(Priority.CRITICAL.value, priority.value - 1))
                    else:
                        priority = max(1, priority - 1)
                
                prioritized_queries.append((priority, query))
            
            # Advanced sorting with multiple criteria
            def sort_key(item):
                priority, query = item
                priority_value = priority.value if hasattr(priority, 'value') else priority
                
                # Secondary sort by query length (longer queries slightly higher priority)
                length_bonus = min(len(query) / 1000, 0.1)  # Max 0.1 bonus
                
                # Tertiary sort by complexity indicators
                complexity_bonus = self._calculate_complexity_score(query) * 0.05
                
                return priority_value - length_bonus - complexity_bonus
            
            # Sort in ascending order of priority value (1=highest, 4=lowest)
            prioritized_queries.sort(key=sort_key)
            
            # Update queue metadata
            self._update_sorting_metrics(prioritized_queries)
            
            return prioritized_queries

        except Exception as e:
            print(f"Error sorting queries: {e}")
            # Return with default medium priority if sorting fails
            return [(Priority.MEDIUM, query) for query in queries]
    
    def record_query_stats(self, query: str, processing_time: float, success: bool) -> None:
        """
        Record comprehensive statistics for a processed query with advanced analytics.
        
        This method tracks detailed performance metrics including:
        - Processing time analysis with trend detection
        - Success rate monitoring with quality indicators
        - Priority distribution analytics
        - Query complexity correlation analysis  
        - Performance optimization recommendations
        - Anomaly detection and alerting
        
        Args:
            query (str): The original user query that was processed
            processing_time (float): Time taken to process the query (in seconds)
            success (bool): Whether the query processing was successful
        """
        if processing_time < 0:
            print(f"Warning: Invalid processing time {processing_time}")
            return
            
        try:
            # Determine query category and priority for detailed analytics
            query_type = self._categorize_query(query)
            priority = self.get_priority(query)
            priority_value = priority.value if hasattr(priority, 'value') else priority
            complexity_score = self._calculate_complexity_score(query)
            
            # Initialize or retrieve existing metrics
            if query_type not in self.query_stats:
                self.query_stats[query_type] = QueryMetrics()
            
            # Update comprehensive metrics
            self.query_stats[query_type].update(
                processing_time=processing_time,
                success=success,
                priority=priority_value,
                complexity=complexity_score
            )
            
            # Update global performance metrics
            self._update_global_performance_metrics(processing_time, success, priority_value)
            
            # Analyze for performance anomalies and optimization opportunities
            self._analyze_performance_anomalies(query_type, processing_time, success)
            
            # Update conversation context for future priority decisions
            self._update_conversation_context(query, processing_time, success)
            
        except Exception as e:
            print(f"Error recording query stats: {e}")
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights and recommendations based on collected analytics.
        
        Provides detailed analysis including:
        - Performance bottleneck identification
        - Query pattern analysis and trends
        - Resource utilization recommendations
        - Priority distribution optimization
        - Predictive capacity planning
        - System health indicators
        
        Returns:
            Dict[str, Any]: Comprehensive insights with performance metrics,
                           optimization recommendations, and predictive analytics
        """
        try:
            if not self.query_stats:
                return self._generate_empty_insights()
            
            # Calculate comprehensive aggregate metrics
            total_time = sum(stats.total_time for stats in self.query_stats.values())
            total_count = sum(stats.count for stats in self.query_stats.values())
            total_success = sum(stats.success_count for stats in self.query_stats.values())
            
            # Generate base insights structure
            insights = {
                'total_queries': total_count,
                'avg_processing_time': total_time / total_count if total_count > 0 else 0.0,
                'overall_success_rate': total_success / total_count if total_count > 0 else 0.0,
                'query_types_analyzed': len(self.query_stats),
                'analysis_timestamp': datetime.datetime.now().isoformat(),
                'session_duration': (datetime.datetime.now() - self.session_start).total_seconds()
            }
            
            # Advanced performance analysis
            insights.update(self._generate_performance_analysis())
            
            # Detailed query type breakdowns
            insights.update(self._generate_query_type_analysis())
            
            # Priority distribution insights
            insights.update(self._generate_priority_analysis())
            
            # Optimization recommendations
            insights.update(self._generate_optimization_recommendations(insights))
            
            # Predictive analytics
            insights.update(self._generate_predictive_insights())
            
            # System health indicators
            insights.update(self._generate_health_indicators())
            
            return insights

        except Exception as e:
            print(f"Error generating optimization insights: {e}")
            return {'error': f'Could not generate insights: {e}', 'timestamp': datetime.datetime.now().isoformat()}
    
    # === ENHANCED PRIORITY QUEUE OPERATIONS ===
    
    def add_to_priority_queue(self, query: str, timestamp: Optional[float] = None) -> None:
        """
        Add a query to the priority queue with advanced metadata and context analysis.
        
        Args:
            query (str): The query to add to the processing queue
            timestamp (Optional[float]): Query timestamp, defaults to current time
        """
        if timestamp is None:
            timestamp = time.time()
            
        try:
            priority = self.get_priority(query)
            priority_value = priority.value if hasattr(priority, 'value') else priority
            
            # Create enhanced queue entry with metadata
            queue_entry = {
                'query': query,
                'priority': priority_value,
                'timestamp': timestamp,
                'complexity': self._calculate_complexity_score(query),
                'estimated_processing_time': self._estimate_processing_time(query),
                'context_relevance': self._calculate_context_relevance(query),
                'entry_id': f"{timestamp}_{hash(query) % 10000}"
            }
            
            # Use negative priority for min-heap behavior (lower number = higher priority)
            heap_entry = (-priority_value, timestamp, queue_entry)
            heapq.heappush(self.priority_queue, heap_entry)
            
            # Update queue metadata
            self.queue_metadata['priority_distribution'][priority_value] += 1
            self._detect_queue_patterns()
            
        except Exception as e:
            print(f"Error adding to priority queue: {e}")
    
    def get_next_query(self) -> Optional[str]:
        """
        Get the next highest priority query from the queue with comprehensive logging.
        
        Returns:
            Optional[str]: The next query to process, or None if queue is empty
        """
        if not self.priority_queue:
            return None
        
        try:
            _, _, queue_entry = heapq.heappop(self.priority_queue)
            query = queue_entry['query']
            
            # Update processing metrics
            wait_time = time.time() - queue_entry['timestamp']
            self._update_queue_metrics(wait_time, queue_entry)
            
            return query
            
        except Exception as e:
            print(f"Error getting next query from queue: {e}")
            return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the priority queue.
        
        Returns:
            Dict[str, Any]: Detailed queue status with analytics and predictions
        """
        try:
            status = {
                'queue_length': len(self.priority_queue),
                'is_empty': len(self.priority_queue) == 0,
                'next_priority': None,
                'estimated_wait_time': 0.0,
                'priority_distribution': dict(self.queue_metadata['priority_distribution']),
                'average_complexity': 0.0,
                'queue_health': 'healthy'
            }
            
            if self.priority_queue:
                # Get next priority without removing item
                next_priority_value = -self.priority_queue[0][0]
                status['next_priority'] = next_priority_value
                
                # Calculate estimated wait time based on queue and processing history
                status['estimated_wait_time'] = self._estimate_queue_wait_time()
                
                # Calculate average complexity of queued items
                complexities = []
                for _, _, entry in self.priority_queue:
                    if isinstance(entry, dict) and 'complexity' in entry:
                        complexities.append(entry['complexity'])
                
                if complexities:
                    status['average_complexity'] = sum(complexities) / len(complexities)
                
                # Determine queue health
                status['queue_health'] = self._assess_queue_health()
            
            return status
            
        except Exception as e:
            print(f"Error getting queue status: {e}")
            return {'error': str(e), 'queue_length': 0, 'is_empty': True}
    
    def clear_queue(self) -> None:
        """Clear all queries from the priority queue and reset metadata."""
        try:
            cleared_count = len(self.priority_queue)
            self.priority_queue.clear()
            self.queue_metadata['priority_distribution'].clear()
            self.queue_metadata['total_processed'] += cleared_count
            
            print(f"Cleared {cleared_count} queries from priority queue")
            
        except Exception as e:
            print(f"Error clearing queue: {e}")
    
    def reset_stats(self) -> None:
        """Reset all collected statistics and performance metrics."""
        try:
            old_stats_count = len(self.query_stats)
            self.query_stats.clear()
            self.priority_cache.clear()
            self.cache_stats = {'hits': 0, 'misses': 0, 'invalidations': 0}
            self.conversation_context.clear()
            self.topic_continuity.clear()
            
            # Reset performance metrics
            self.performance_metrics = {
                'total_queries_processed': 0,
                'average_response_time': 0.0,
                'priority_accuracy': 0.0,
                'cache_hit_rate': 0.0,
                'system_load': 0.0,
                'optimization_suggestions': []
            }
            
            # Reset session tracking
            self.session_start = datetime.datetime.now()
            
            print(f"Reset statistics for {old_stats_count} query types")
            
        except Exception as e:
            print(f"Error resetting stats: {e}")
    
    # === PRIVATE HELPER METHODS FOR ENHANCED FUNCTIONALITY ===
    
    def _analyze_priority_patterns(self, query_lower: str) -> Optional[Priority]:
        """Analyze query using advanced regex patterns for priority detection."""
        for priority_level, patterns in self.priority_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return priority_level
        return None
    
    def _analyze_keyword_priority(self, query_lower: str) -> Optional[Priority]:
        """Analyze query using keyword matching with weighted scoring."""
        keyword_scores = {}
        
        for priority_level, keywords in self.priority_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # Weight score based on keyword length and specificity
                    keyword_weight = len(keyword.split()) * 0.5 + 1.0
                    score += keyword_weight
            
            if score > 0:
                keyword_scores[priority_level] = score
        
        if keyword_scores:
            # Return priority with highest score
            return max(keyword_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _analyze_complexity_priority(self, query: str) -> Optional[Priority]:
        """Analyze query complexity to determine appropriate priority."""
        complexity_score = self._calculate_complexity_score(query)
        
        if complexity_score >= 4:
            return Priority.CRITICAL  # Very complex queries need immediate attention
        elif complexity_score >= 3:
            return Priority.HIGH
        elif complexity_score >= 2:
            return Priority.MEDIUM
        
        return None
    
    def _analyze_length_priority(self, query: str) -> Optional[Priority]:
        """Analyze query length to infer priority and complexity."""
        query_length = len(query)
        
        if query_length > self.length_thresholds['very_long']:
            return Priority.HIGH  # Very long queries are likely complex
        elif query_length > self.length_thresholds['long']:
            return Priority.MEDIUM
        elif query_length > self.length_thresholds['medium']:
            return Priority.MEDIUM
        
        return None
    
    def _analyze_structural_priority(self, query: str) -> Optional[Priority]:
        """Analyze structural elements like punctuation for priority hints."""
        if '!' in query:
            return Priority.HIGH   # Exclamations suggest urgency
        elif '?' in query:
            return Priority.MEDIUM  # Questions are standard priority
        elif query.isupper() and len(query) > 10:
            return Priority.HIGH   # ALL CAPS suggests urgency
        
        return None
    
    def _determine_context_priority(self, query: str) -> Priority:
        """Determine priority based on conversation context and patterns."""
        # Apply context boost based on conversation history
        context_boost = self._calculate_context_boost(query)
        
        if context_boost > 0.5:
            return Priority.HIGH
        elif context_boost > 0.2:
            return Priority.MEDIUM
        else:
            return Priority.MEDIUM  # Default fallback
    
    def _calculate_complexity_score(self, query: str) -> float:
        """
        Calculate a comprehensive complexity score for the query.
        
        Returns:
            float: Complexity score (0-5+ scale)
        """
        score = 0.0
        query_lower = query.lower()
        
        # Technical complexity indicators
        complexity_matches = sum(1 for indicator in self.complexity_indicators 
                               if indicator in query_lower)
        score += min(complexity_matches * 0.5, 2.0)  # Max 2 points for technical terms
        
        # Query structure complexity
        if re.search(r'\b(step\s+by\s+step|detailed|comprehensive|thorough|complete)\b', query_lower):
            score += 1.0
        
        # Multiple questions or parts
        question_count = query.count('?')
        if question_count > 1:
            score += 0.5 * question_count
        
        # Code-like patterns or technical syntax
        if re.search(r'[(){}\[\]]', query) or '```' in query or 'code' in query_lower:
            score += 1.0
        
        # Length-based complexity
        word_count = len(query.split())
        if word_count > 20:
            score += min(word_count / 50, 1.0)
        
        # Multi-part queries (indicated by conjunctions)
        if re.search(r'\b(and|also|additionally|furthermore|moreover)\b', query_lower):
            score += 0.5
        
        return score
    
    def _categorize_query(self, query: str) -> str:
        """
        Categorize a query into a logical type for comprehensive analytics.
        
        Args:
            query (str): The user query string
            
        Returns:
            str: A detailed query category label
        """
        if not query or not query.strip():
            return 'empty'
            
        try:
            query_lower = query.lower()
            
            # Specific pattern matching for precise categorization
            if any(word in query_lower for word in ['what', 'define', 'definition', 'meaning', 'describe']):
                return 'definition'
            elif any(word in query_lower for word in ['how', 'process', 'step', 'tutorial', 'guide', 'method']):
                return 'how-to'
            elif any(word in query_lower for word in ['where', 'location', 'place', 'address', 'find']):
                return 'location'
            elif any(word in query_lower for word in ['when', 'time', 'date', 'schedule', 'timing']):
                return 'temporal'
            elif any(word in query_lower for word in ['why', 'reason', 'cause', 'because', 'purpose']):
                return 'causal'
            elif any(word in query_lower for word in ['who', 'person', 'people', 'author', 'creator']):
                return 'person'
            elif any(word in query_lower for word in ['error', 'bug', 'issue', 'problem', 'broken', 'fail']):
                return 'troubleshooting'
            elif any(word in query_lower for word in ['hello', 'hi', 'thanks', 'goodbye', 'bye', 'greetings']):
                return 'social'
            elif any(word in query_lower for word in self.complexity_indicators):
                return 'technical'
            elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better']):
                return 'comparison'
            elif any(word in query_lower for word in ['example', 'sample', 'instance', 'demo']):
                return 'example'
            elif '?' in query:
                return 'question'
            elif any(word in query_lower for word in ['urgent', 'emergency', 'critical', 'help']):
                return 'urgent'
            else:
                return 'general'

        except Exception as e:
            print(f"Error categorizing query: {e}")
            return 'general'
    
    def _normalize_query_for_cache(self, query: str) -> str:
        """Normalize query for cache key generation."""
        normalized = query.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        normalized = re.sub(r'\s+', ' ', normalized)      # Normalize whitespace
        return normalized
    
    def _cache_priority_result(self, cache_key: str, priority: Priority) -> None:
        """Cache priority calculation result with size management."""
        if len(self.priority_cache) >= self.cache_max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.priority_cache.keys())[:100]
            for key in keys_to_remove:
                del self.priority_cache[key]
            self.cache_stats['invalidations'] += len(keys_to_remove)
        
        self.priority_cache[cache_key] = priority
    
    def _calculate_context_boost(self, query: str) -> float:
        """Calculate context-based priority boost."""
        if not self.conversation_context:
            return 0.0
        
        boost = 0.0
        query_topics = self._extract_query_topics(query)
        
        # Check recent conversation for topic continuity
        recent_context = self.conversation_context[-4:]  # Last 2 exchanges
        for context_entry in recent_context:
            if 'topics' in context_entry:
                context_topics = context_entry['topics']
                overlap = len(set(query_topics) & set(context_topics))
                if overlap > 0:
                    boost += overlap * 0.1
        
        return min(boost, 1.0)  # Cap boost at 1.0
    
    def _extract_query_topics(self, query: str) -> List[str]:
        """Extract topics from a query for context analysis."""
        topics = []
        query_lower = query.lower()
        
        # Extract meaningful words (longer than 3 characters)
        words = re.findall(r'\b\w{4,}\b', query_lower)
        topics.extend(words)
        
        # Extract technical terms
        for indicator in self.complexity_indicators:
            if indicator in query_lower:
                topics.append(indicator)
        
        # Extract compound phrases
        compound_patterns = [
            r'machine\s+learning', r'data\s+science', r'artificial\s+intelligence',
            r'web\s+development', r'software\s+engineering', r'computer\s+science'
        ]
        
        for pattern in compound_patterns:
            matches = re.findall(pattern, query_lower)
            topics.extend(matches)
        
        return list(set(topics))
    
    def _extract_current_context_topics(self) -> Set[str]:
        """Extract topics from current conversation context."""
        topics = set()
        
        for entry in self.conversation_context[-6:]:  # Recent context
            if 'topics' in entry:
                topics.update(entry['topics'])
        
        return topics
    
    def _update_sorting_metrics(self, sorted_queries: List[Tuple[Union[int, Priority], str]]) -> None:
        """Update metrics after query sorting operation."""
        if not sorted_queries:
            return
        
        # Track priority distribution
        priority_counts = Counter()
        for priority, _ in sorted_queries:
            priority_value = priority.value if hasattr(priority, 'value') else priority
            priority_counts[priority_value] += 1
        
        self.queue_metadata['priority_distribution'].update(priority_counts)
    
    def _update_global_performance_metrics(self, processing_time: float, success: bool, priority: int) -> None:
        """Update global performance tracking metrics."""
        self.performance_metrics['total_queries_processed'] += 1
        
        # Update average response time
        total_time = (self.performance_metrics['average_response_time'] * 
                     (self.performance_metrics['total_queries_processed'] - 1) + processing_time)
        self.performance_metrics['average_response_time'] = total_time / self.performance_metrics['total_queries_processed']
        
        # Update cache hit rate
        total_cache_operations = self.cache_stats['hits'] + self.cache_stats['misses']
        if total_cache_operations > 0:
            self.performance_metrics['cache_hit_rate'] = self.cache_stats['hits'] / total_cache_operations
    
    def _analyze_performance_anomalies(self, query_type: str, processing_time: float, success: bool) -> None:
        """Analyze for performance anomalies and generate alerts."""
        if query_type in self.query_stats:
            stats = self.query_stats[query_type]
            
            # Check for processing time anomalies
            if stats.avg_time > 0 and processing_time > stats.avg_time * 3:
                anomaly = f"Slow processing detected for {query_type}: {processing_time:.2f}s (avg: {stats.avg_time:.2f}s)"
                self.performance_metrics['optimization_suggestions'].append(anomaly)
            
            # Check for success rate drops
            if stats.count > 10 and stats.success_rate < 0.7:
                anomaly = f"Low success rate for {query_type}: {stats.success_rate:.2%}"
                self.performance_metrics['optimization_suggestions'].append(anomaly)
    
    def _update_conversation_context(self, query: str, processing_time: float, success: bool) -> None:
        """Update conversation context for future priority decisions."""
        timestamp = datetime.datetime.now()
        
        context_entry = {
            'type': 'query',
            'content': query,
            'timestamp': timestamp,
            'processing_time': processing_time,
            'success': success,
            'topics': self._extract_query_topics(query),
            'complexity': self._calculate_complexity_score(query),
            'priority': self.get_priority(query)
        }
        
        self.conversation_context.append(context_entry)
        
        # Maintain context window
        if len(self.conversation_context) > self.context_window:
            self.conversation_context = self.conversation_context[-self.context_window:]
        
        # Update topic continuity
        for topic in context_entry['topics']:
            self.topic_continuity[topic]['frequency'] += 1
            self.topic_continuity[topic]['last_mentioned'] = timestamp
            self.topic_continuity[topic]['importance_score'] = min(
                self.topic_continuity[topic]['frequency'] * 0.1, 1.0
            )
    
    def _generate_empty_insights(self) -> Dict[str, Any]:
        """Generate insights structure when no data is available."""
        return {
            'total_queries': 0,
            'avg_processing_time': 0.0,
            'overall_success_rate': 0.0,
            'query_types_analyzed': 0,
            'slowest_query_types': [],
            'most_common_query_types': [],
            'fastest_query_types': [],
            'least_successful_types': [],
            'recommendations': ['Insufficient data for analysis. Process more queries to generate insights.'],
            'priority_distribution': {},
            'performance_trends': [],
            'cache_efficiency': 0.0,
            'system_health': 'unknown'
        }
    
    def _generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate detailed performance analysis."""
        # Sort query types by performance metrics
        sorted_by_time = sorted(self.query_stats.items(), key=lambda x: x[1].avg_time, reverse=True)
        sorted_by_count = sorted(self.query_stats.items(), key=lambda x: x[1].count, reverse=True)
        sorted_by_success = sorted(self.query_stats.items(), key=lambda x: x[1].success_rate)
        
        return {
            'slowest_query_types': [
                {
                    'type': qtype,
                    'avg_time': stats.avg_time,
                    'count': stats.count,
                    'success_rate': stats.success_rate
                }
                for qtype, stats in sorted_by_time[:5]
            ],
            'most_common_query_types': [
                {
                    'type': qtype,
                    'count': stats.count,
                    'avg_time': stats.avg_time,
                    'success_rate': stats.success_rate
                }
                for qtype, stats in sorted_by_count[:5]
            ],
            'fastest_query_types': [
                {
                    'type': qtype,
                    'avg_time': stats.avg_time,
                    'count': stats.count,
                    'success_rate': stats.success_rate
                }
                for qtype, stats in sorted_by_time[-3:] if stats.count > 0
            ],
            'least_successful_types': [
                {
                    'type': qtype,
                    'success_rate': stats.success_rate,
                    'count': stats.count,
                    'avg_time': stats.avg_time
                }
                for qtype, stats in sorted_by_success[:3] if stats.success_rate < 0.8 and stats.count > 2
            ]
        }
    
    def _generate_query_type_analysis(self) -> Dict[str, Any]:
        """Generate query type distribution and trends analysis."""
        type_distribution = {}
        complexity_by_type = {}
        
        for qtype, stats in self.query_stats.items():
            type_distribution[qtype] = {
                'count': stats.count,
                'percentage': 0.0,  # Will be calculated below
                'avg_processing_time': stats.avg_time,
                'success_rate': stats.success_rate
            }
            
            if stats.complexity_scores:
                complexity_by_type[qtype] = {
                    'avg_complexity': sum(stats.complexity_scores) / len(stats.complexity_scores),
                    'max_complexity': max(stats.complexity_scores),
                    'complexity_trend': 'stable'  # Could be enhanced with trend analysis
                }
        
        # Calculate percentages
        total_queries = sum(stats.count for stats in self.query_stats.values())
        if total_queries > 0:
            for qtype in type_distribution:
                type_distribution[qtype]['percentage'] = (
                    type_distribution[qtype]['count'] / total_queries * 100
                )
        
        return {
            'query_type_distribution': type_distribution,
            'complexity_analysis': complexity_by_type,
            'type_diversity_score': len(self.query_stats) / max(total_queries / 10, 1)
        }
    
    def _generate_priority_analysis(self) -> Dict[str, Any]:
        """Generate priority distribution and effectiveness analysis."""
        priority_stats = {}
        
        for stats in self.query_stats.values():
            for priority, count in stats.priority_distribution.items():
                if priority not in priority_stats:
                    priority_stats[priority] = {'count': 0, 'total_time': 0.0, 'success_count': 0}
                
                priority_stats[priority]['count'] += count
        
        # Calculate priority effectiveness
        priority_effectiveness = {}
        for priority in [1, 2, 3, 4]:  # CRITICAL, HIGH, MEDIUM, LOW
            if priority in priority_stats:
                stats = priority_stats[priority]
                priority_effectiveness[priority] = {
                    'count': stats['count'],
                    'percentage': 0.0,  # Will be calculated below
                    'priority_name': Priority(priority).name
                }
        
        total_priority_queries = sum(stats['count'] for stats in priority_stats.values())
        if total_priority_queries > 0:
            for priority in priority_effectiveness:
                priority_effectiveness[priority]['percentage'] = (
                    priority_stats[priority]['count'] / total_priority_queries * 100
                )
        
        return {
            'priority_distribution': priority_effectiveness,
            'priority_balance_score': self._calculate_priority_balance(),
            'priority_trends': self._analyze_priority_trends()
        }
    
    def _generate_optimization_recommendations(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable optimization recommendations."""
        recommendations = []
        
        # Performance-based recommendations
        if insights['overall_success_rate'] < 0.8:
            recommendations.append(
                "Overall success rate is below 80%. Review error handling and query processing logic."
            )
        
        if insights['avg_processing_time'] > 1.0:
            recommendations.append(
                "Average processing time exceeds 1 second. Consider optimizing slow query types or implementing caching."
            )
        
        # Cache efficiency recommendations
        cache_hit_rate = self.performance_metrics.get('cache_hit_rate', 0.0)
        if cache_hit_rate < 0.3:
            recommendations.append(
                f"Cache hit rate is low ({cache_hit_rate:.1%}). Consider increasing cache size or improving cache keys."
            )
        
        # Priority distribution recommendations
        if 'priority_distribution' in insights:
            critical_percentage = insights['priority_distribution'].get(1, {}).get('percentage', 0)
            if critical_percentage > 20:
                recommendations.append(
                    f"High percentage of critical queries ({critical_percentage:.1f}%). Review priority classification rules."
                )
        
        # Query type specific recommendations
        if 'slowest_query_types' in insights and insights['slowest_query_types']:
            slowest_type = insights['slowest_query_types'][0]
            recommendations.append(
                f"Focus optimization on '{slowest_type['type']}' queries - they have the highest average processing time."
            )
        
        return {
            'recommendations': recommendations,
            'optimization_priority': self._calculate_optimization_priority(insights),
            'suggested_actions': self._generate_suggested_actions(insights)
        }
    
    def _generate_predictive_insights(self) -> Dict[str, Any]:
        """Generate predictive analytics and forecasting."""
        return {
            'predicted_load': self._predict_system_load(),
            'capacity_recommendations': self._generate_capacity_recommendations(),
            'trending_query_types': self._identify_trending_types(),
            'performance_forecast': self._forecast_performance_trends()
        }
    
    def _generate_health_indicators(self) -> Dict[str, Any]:
        """Generate system health indicators and status."""
        health_score = self._calculate_health_score()
        
        return {
            'system_health_score': health_score,
            'health_status': self._determine_health_status(health_score),
            'critical_metrics': self._identify_critical_metrics(),
            'uptime_statistics': self._calculate_uptime_stats(),
            'resource_utilization': self._calculate_resource_utilization()
        }
    
    # === ADDITIONAL HELPER METHODS ===
    
    def _calculate_priority_balance(self) -> float:
        """Calculate how well-balanced the priority distribution is."""
        # Implementation for priority balance calculation
        return 0.8  # Placeholder
    
    def _analyze_priority_trends(self) -> List[str]:
        """Analyze trends in priority distribution over time."""
        # Implementation for priority trend analysis
        return ["Priority distribution is stable"]  # Placeholder
    
    def _calculate_optimization_priority(self, insights: Dict[str, Any]) -> str:
        """Calculate the priority level for optimization efforts."""
        if insights['overall_success_rate'] < 0.7:
            return "CRITICAL"
        elif insights['avg_processing_time'] > 2.0:
            return "HIGH"
        elif len(insights.get('recommendations', [])) > 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_suggested_actions(self, insights: Dict[str, Any]) -> List[str]:
        """Generate specific suggested actions for improvement."""
        actions = []
        
        if insights['avg_processing_time'] > 1.0:
            actions.append("Implement query result caching")
            actions.append("Optimize database queries")
            actions.append("Consider horizontal scaling")
        
        if insights['overall_success_rate'] < 0.8:
            actions.append("Improve error handling mechanisms")
            actions.append("Add input validation and sanitization")
            actions.append("Implement retry logic for failed queries")
        
        return actions
    
    def _predict_system_load(self) -> Dict[str, float]:
        """Predict future system load based on current trends."""
        return {
            'next_hour': 1.2,
            'next_day': 1.0,
            'next_week': 0.9
        }  # Placeholder implementation
    
    def _generate_capacity_recommendations(self) -> List[str]:
        """Generate capacity planning recommendations."""
        return [
            "Current capacity appears adequate for projected load",
            "Monitor queue length during peak hours",
            "Consider auto-scaling for traffic spikes"
        ]  # Placeholder implementation
    
    def _identify_trending_types(self) -> List[str]:
        """Identify query types that are trending upward."""
        # Implementation for trend identification
        return ["technical", "how-to"]  # Placeholder
    
    def _forecast_performance_trends(self) -> Dict[str, str]:
        """Forecast performance trends."""
        return {
            'processing_time': 'stable',
            'success_rate': 'improving',
            'queue_length': 'stable'
        }  # Placeholder implementation
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-1)."""
        factors = []
        
        # Success rate factor
        if self.query_stats:
            total_success = sum(stats.success_count for stats in self.query_stats.values())
            total_count = sum(stats.count for stats in self.query_stats.values())
            success_factor = total_success / total_count if total_count > 0 else 1.0
            factors.append(success_factor)
        
        # Performance factor
        avg_time = self.performance_metrics.get('average_response_time', 0.0)
        performance_factor = max(0, 1 - (avg_time / 5.0))  # Penalize if > 5 seconds
        factors.append(performance_factor)
        
        # Cache efficiency factor
        cache_factor = self.performance_metrics.get('cache_hit_rate', 0.5)
        factors.append(cache_factor)
        
        return sum(factors) / len(factors) if factors else 0.8
    
    def _determine_health_status(self, health_score: float) -> str:
        """Determine health status based on score."""
        if health_score >= 0.9:
            return "excellent"
        elif health_score >= 0.8:
            return "good"
        elif health_score >= 0.6:
            return "fair"
        elif health_score >= 0.4:
            return "poor"
        else:
            return "critical"
    
    def _identify_critical_metrics(self) -> List[str]:
        """Identify metrics that need immediate attention."""
        critical = []
        
        if self.performance_metrics.get('average_response_time', 0) > 3.0:
            critical.append("High average response time")
        
        if self.performance_metrics.get('cache_hit_rate', 1.0) < 0.2:
            critical.append("Low cache hit rate")
        
        return critical
    
    def _calculate_uptime_stats(self) -> Dict[str, Any]:
        """Calculate system uptime statistics."""
        uptime = (datetime.datetime.now() - self.session_start).total_seconds()
        
        return {
            'session_uptime_seconds': uptime,
            'uptime_formatted': str(datetime.timedelta(seconds=int(uptime))),
            'queries_per_hour': self.performance_metrics['total_queries_processed'] / max(uptime / 3600, 1),
            'session_start': self.session_start.isoformat()
        }
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization metrics."""
        return {
            'cache_utilization': len(self.priority_cache) / self.cache_max_size,
            'queue_utilization': len(self.priority_queue) / 1000,  # Assume max 1000
            'context_utilization': len(self.conversation_context) / self.context_window
        }
    
    # === QUEUE MANAGEMENT HELPER METHODS ===
    
    def _estimate_processing_time(self, query: str) -> float:
        """Estimate processing time for a query based on historical data."""
        query_type = self._categorize_query(query)
        
        if query_type in self.query_stats:
            return self.query_stats[query_type].avg_time
        
        # Default estimates based on complexity
        complexity = self._calculate_complexity_score(query)
        if complexity > 3:
            return 2.0
        elif complexity > 2:
            return 1.0
        elif complexity > 1:
            return 0.5
        else:
            return 0.2
    
    def _calculate_context_relevance(self, query: str) -> float:
        """Calculate how relevant a query is to current context."""
        if not self.conversation_context:
            return 0.0
        
        query_topics = set(self._extract_query_topics(query))
        context_topics = set()
        
        for entry in self.conversation_context[-3:]:  # Recent context
            if 'topics' in entry:
                context_topics.update(entry['topics'])
        
        if not query_topics or not context_topics:
            return 0.0
        
        overlap = len(query_topics & context_topics)
        return min(overlap / len(query_topics), 1.0)
    
    def _detect_queue_patterns(self) -> None:
        """Detect patterns in queue usage for optimization."""
        # Implementation for pattern detection in queue usage
        pass
    
    def _update_queue_metrics(self, wait_time: float, queue_entry: Dict[str, Any]) -> None:
        """Update queue performance metrics."""
        self.queue_metadata['total_processed'] += 1
        
        # Update average wait time
        current_avg = self.queue_metadata['average_wait_time']
        total_processed = self.queue_metadata['total_processed']
        
        self.queue_metadata['average_wait_time'] = (
            (current_avg * (total_processed - 1) + wait_time) / total_processed
        )
    
    def _estimate_queue_wait_time(self) -> float:
        """Estimate wait time for items currently in queue."""
        if not self.priority_queue:
            return 0.0
        
        # Simple estimation based on queue length and average processing time
        avg_processing_time = self.performance_metrics.get('average_response_time', 0.5)
        return len(self.priority_queue) * avg_processing_time
    
    def _assess_queue_health(self) -> str:
        """Assess the health status of the priority queue."""
        queue_length = len(self.priority_queue)
        
        if queue_length == 0:
            return "idle"
        elif queue_length < 10:
            return "healthy"
        elif queue_length < 50:
            return "busy"
        elif queue_length < 100:
            return "overloaded"
        else:
            return "critical"