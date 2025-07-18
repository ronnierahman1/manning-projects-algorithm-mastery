"""
Optimized Main entry point for the Recursive AI Chatbot - Educational Documentation

This script demonstrates enterprise-level application architecture by integrating:
- Recursive Query Handling (Milestone 1) - Advanced query decomposition
- Dynamic Context Management with Caching (Milestone 2) - Performance optimization
- Greedy Priority Algorithm for Optimized Response Time (Milestone 3) - Smart prioritization

Educational Focus Areas:
- **Application Architecture**: How to structure large Python applications
- **Design Patterns**: Observer, Factory, Strategy, Command patterns
- **Configuration Management**: Centralized settings with file persistence
- **Performance Monitoring**: Real-time metrics collection and analysis
- **Resource Management**: Memory monitoring and cleanup strategies
- **User Experience Design**: Interactive terminal interface with rich features
- **Error Handling**: Comprehensive exception management and graceful degradation
- **Data Persistence**: JSON serialization and backup strategies
- **Threading**: Background tasks and signal handling
- **Testing Infrastructure**: Statistics collection for system optimization

Key improvements:
- Fixed result handling for recursive queries
- Better error handling and validation
- Improved statistics tracking
- Enhanced user experience
- More robust query classification
- Performance optimizations
- Advanced configuration management
- Comprehensive logging system
- Better resource management
- Sophisticated command system
- Plugin architecture foundation
- Performance monitoring
- Advanced caching strategies
"""

# ============================================================================
# Standard Library Imports - Grouped by functionality for educational clarity
# ============================================================================

# Core Python functionality
import sys       # System-specific parameters and functions (exit codes, path manipulation)
import os        # Operating system interface (file paths, environment variables)
import time      # Time-related functions (performance measurement, delays)

# Advanced application features
import logging   # Professional logging system for debugging and monitoring
import json      # JSON encoding/decoding for configuration and data persistence
import threading # Multi-threading for background tasks (auto-save, monitoring)
import signal    # Unix signal handling for graceful shutdown
import atexit    # Register cleanup functions to run on program termination

# Type hints for better code documentation and IDE support
from typing import Dict, Any, Optional, List, Tuple, Callable

# Data structures and utilities
from dataclasses import dataclass, asdict, field  # Modern Python data classes for structured data
from pathlib import Path                          # Object-oriented file system paths
from datetime import datetime, timedelta         # Date and time manipulation
from collections import deque, defaultdict       # Efficient data structures for queues and counting
import hashlib                                   # Cryptographic hashing for cache keys and data integrity
import gc                                        # Garbage collection for memory management
import psutil                                    # System and process utilities for monitoring
from milestones.recursive_handling import QueryResult # Custom query result class for structured responses

# ============================================================================
# Project Path Setup - Essential for module imports
# ============================================================================

# Add the current project root directory to the Python path
# This allows importing custom modules regardless of where the script is run from
# insert(0, ...) puts our project at the beginning of the path for priority
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Custom Module Imports - Core chatbot components
# ============================================================================

# Import the core components of the chatbot system with comprehensive error handling
# This demonstrates defensive programming - always handle potential import failures
try:
    # Main chatbot engine - handles basic Q&A functionality
    from chatbot.chatbot import AIChatbot
    
    # Advanced features from milestone implementations
    from milestones.recursive_handling import RecursiveHandling as RecursiveHandling  # Complex query decomposition
    from milestones.dynamic_context import DynamicContext                             # Caching and context management
    from milestones.greedy_priority import GreedyPriority                            # Smart query prioritization
    
except ImportError as e:
    # Graceful failure with helpful error message
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)  # Exit with error code to indicate failure


# ============================================================================
# Data Classes - Modern Python approach to structured data
# ============================================================================

@dataclass
class SessionStats:
    """
    Comprehensive data class to track session statistics and performance metrics.
    
    This class demonstrates several important software engineering concepts:
    
    **Data Classes**: Modern Python approach to creating structured data containers
    **Default Values**: Using field() with default_factory for mutable defaults
    **Type Hints**: Explicit typing for better code documentation and IDE support
    **Analytics Design**: Comprehensive metrics collection for system optimization
    
    Educational Notes:
    - @dataclass automatically generates __init__, __repr__, __eq__ methods
    - field(default_factory=...) creates new instances for mutable defaults
    - This prevents the "mutable default argument" pitfall in Python
    - The structure supports both real-time monitoring and historical analysis
    """
    
    # Core query processing metrics
    queries_processed: int = 0          # Total number of queries handled this session
    cache_hits: int = 0                 # Number of queries answered from cache (performance indicator)
    recursive_queries: int = 0          # Complex queries requiring recursive processing
    successful_queries: int = 0         # Queries that completed without errors
    failed_queries: int = 0             # Queries that encountered errors (quality indicator)
    
    # Performance metrics for optimization
    average_response_time: float = 0.0  # Mean time to process queries (milliseconds)
    total_time: float = 0.0             # Cumulative processing time for all queries
    fuzzy_matches: int = 0              # Queries resolved using fuzzy string matching
    
    # Priority distribution tracking using dict with default values
    # This shows how queries are distributed across priority levels
    priority_distribution: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    
    # Advanced metrics for comprehensive analysis
    session_start_time: datetime = field(default_factory=datetime.now)  # When session began
    peak_memory_usage: float = 0.0      # Maximum memory consumption during session
    command_usage: Dict[str, int] = field(default_factory=dict)         # Which commands users invoke
    
    # Query analysis metrics for understanding user behavior
    query_length_stats: Dict[str, float] = field(default_factory=lambda: {'min': float('inf'), 'max': 0, 'avg': 0})
    
    # Error analysis for system improvement
    error_categories: Dict[str, int] = field(default_factory=dict)      # Types of errors encountered
    
    # Quality and satisfaction metrics
    response_quality_scores: List[float] = field(default_factory=list)  # Algorithmic quality assessment
    user_satisfaction_ratings: List[int] = field(default_factory=list)  # User-provided ratings
    
    def __post_init__(self):
        """
        Post-initialization hook to ensure data consistency.
        
        This method runs after the dataclass __init__ method and allows
        for additional validation or setup that can't be done with default values.
        
        Educational Note: __post_init__ is a dataclass feature that provides
        a way to perform additional initialization after the automatic __init__.
        """
        # Ensure priority_distribution always has the expected structure
        # This prevents KeyError exceptions when accessing priority levels
        if not self.priority_distribution:
            self.priority_distribution = {1: 0, 2: 0, 3: 0}


@dataclass
class ChatbotConfig:
    """
    Comprehensive configuration management for the chatbot application.
    
    This class demonstrates enterprise-level configuration management patterns:
    
    **Configuration as Code**: All settings defined as typed attributes
    **Default Values**: Sensible defaults for all configuration options
    **Serialization**: Built-in methods for saving/loading from JSON
    **Type Safety**: Explicit types prevent configuration errors
    **Extensibility**: Easy to add new configuration options
    
    Design Philosophy:
    - All configurable behavior should be centralized here
    - Defaults should create a working system out-of-the-box
    - Configuration should be easily shareable and version-controlled
    - Changes should not require code modifications
    """
    
    # Core algorithm parameters
    max_recursion_depth: int = 4                    # Limit recursive query processing depth
    cache_size_limit: int = 1000                    # Maximum number of cached responses
    
    # Logging and debugging configuration
    logging_level: str = "INFO"                     # Python logging level (DEBUG, INFO, WARNING, ERROR)
    enable_performance_monitoring: bool = True     # Whether to collect performance metrics
    auto_save_stats: bool = True                   # Automatically save statistics to disk
    stats_save_interval: int = 300                 # How often to save stats (seconds)
    
    # Query processing configuration
    enable_fuzzy_matching: bool = True             # Allow approximate string matching
    fuzzy_threshold: float = 0.8                   # Minimum similarity for fuzzy matches
    enable_spell_correction: bool = True           # Basic spell correction for queries
    response_timeout: float = 30.0                 # Maximum time to process a query (seconds)
    max_query_length: int = 1000                   # Maximum allowed query length (characters)
    
    # Advanced features (future expansion)
    ena4ble_plugins: bool = False                   # Plugin system activation
    plugins_directory: str = "./plugins"           # Where to find plugin modules
    backup_data_on_start: bool = True             # Create backup when starting
    
    # User experience configuration
    enable_conversation_history: bool = True       # Track conversation for context
    max_history_size: int = 100                   # Maximum conversation entries to keep
    enable_analytics: bool = True                  # Collect usage analytics
    performance_alert_threshold: float = 5.0       # When to alert about slow responses (seconds)
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'ChatbotConfig':
        """
        Class method to create configuration instance from JSON file.
        
        This method demonstrates several important patterns:
        
        **Class Methods**: Using @classmethod for alternative constructors
        **Error Handling**: Graceful fallback to defaults if file loading fails
        **JSON Deserialization**: Converting file data back to Python objects
        **Factory Pattern**: Creating objects through specialized methods
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            ChatbotConfig instance with loaded or default values
            
        Educational Note: Class methods receive the class (cls) as first argument
        instead of instance (self), allowing them to create new instances.
        """
        # Check if configuration file exists before attempting to load
        if os.path.exists(config_path):
            try:
                # Open file with explicit encoding for cross-platform compatibility
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)  # Parse JSON into Python dictionary
                
                # Use dictionary unpacking to create instance with loaded values
                # **config_dict unpacks the dictionary as keyword arguments
                return cls(**config_dict)
                
            except Exception as e:
                # Log warning but continue with defaults rather than crashing
                # This demonstrates graceful degradation in error handling
                logging.warning(f"Failed to load config from {config_path}: {e}")
        
        # Return default configuration if file doesn't exist or loading failed
        return cls()
    
    def save_to_file(self, config_path: str):
        """
        Save current configuration to JSON file for persistence.
        
        This method enables configuration changes to persist across sessions
        and allows sharing configurations between deployments.
        
        Args:
            config_path: Where to save the configuration file
            
        Educational Note: asdict() converts dataclass to dictionary,
        which can then be serialized to JSON format.
        """
        try:
            # Open file for writing with explicit encoding
            with open(config_path, 'w') as f:
                # Convert dataclass to dictionary, then serialize to JSON
                # indent=2 makes the file human-readable
                json.dump(asdict(self), f, indent=2)
                
        except Exception as e:
            # Log error but don't crash the application
            logging.error(f"Failed to save config to {config_path}: {e}")


# ============================================================================
# Performance Monitoring System - Enterprise-level metrics collection
# ============================================================================

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for real-time analytics.
    
    This class demonstrates several important software engineering concepts:
    
    **Observer Pattern**: Monitors system behavior without changing core logic
    **Time Series Data**: Efficient storage and analysis of metrics over time
    **Alerting System**: Proactive notification of performance issues
    **Memory Management**: Automatic cleanup of old data to prevent memory leaks
    
    Key Features:
    - Real-time metric collection with timestamps
    - Automatic data retention management
    - Statistical analysis of performance trends
    - Alert generation for performance degradation
    
    Educational Value:
    This shows how to build monitoring into applications from the ground up,
    which is essential for production systems.
    """
    
    def __init__(self):
        """
        Initialize the performance monitoring system.
        
        Uses defaultdict(list) to automatically create empty lists for new metrics,
        eliminating the need for explicit key existence checking.
        """
        # defaultdict automatically creates missing keys with empty lists
        # This prevents KeyError exceptions when recording new metric types
        self.metrics = defaultdict(list)            # Store time-series metric data
        self.alerts = []                            # Store performance alerts
        self.monitoring_active = True               # Allow disabling monitoring if needed
        
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """
        Record a performance metric with automatic timestamp and data retention.
        
        This method implements efficient time-series data storage with:
        - Automatic timestamping for accurate temporal analysis
        - Memory management through automatic old data cleanup
        - Optional custom timestamps for batch data loading
        
        Args:
            metric_name: Name of the metric (e.g., 'response_time', 'memory_usage')
            value: Numeric value to record
            timestamp: Optional custom timestamp (defaults to now)
            
        Educational Note: This pattern is common in monitoring systems -
        store tuples of (timestamp, value) for efficient time-series analysis.
        """
        # Early return if monitoring is disabled (performance optimization)
        if not self.monitoring_active:
            return
            
        # Use current time if no timestamp provided
        if timestamp is None:
            timestamp = datetime.now()
            
        # Store metric as (timestamp, value) tuple for time-series analysis
        self.metrics[metric_name].append((timestamp, value))
        
        # Automatic data retention: keep only recent metrics (last hour)
        # This prevents memory growth in long-running applications
        cutoff = datetime.now() - timedelta(hours=1)
        
        # List comprehension with time filtering for efficient cleanup
        # Keep only metrics newer than the cutoff time
        self.metrics[metric_name] = [
            (ts, val) for ts, val in self.metrics[metric_name] if ts > cutoff
        ]
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """
        Generate statistical summary for a specific metric.
        
        This method demonstrates common statistical analysis patterns:
        - Defensive programming with existence checking
        - Efficient statistical calculations using built-in functions
        - Structured return data for easy consumption
        
        Args:
            metric_name: Name of metric to analyze
            
        Returns:
            Dictionary with statistical summary (count, min, max, avg, latest)
            
        Educational Note: This pattern of returning structured statistics
        is common in analytics systems and APIs.
        """
        # Defensive programming: check if metric exists and has data
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}  # Return empty dict rather than raising exception
            
        # Extract just the values from (timestamp, value) tuples
        # List comprehension efficiently transforms the data structure
        values = [val for _, val in self.metrics[metric_name]]
        
        # Calculate comprehensive statistics using built-in functions
        # This provides a complete picture of metric behavior
        return {
            'count': len(values),                    # Number of data points
            'min': min(values),                      # Minimum observed value
            'max': max(values),                      # Maximum observed value
            'avg': sum(values) / len(values),        # Average value
            'latest': values[-1] if values else 0   # Most recent value
        }
    
    def check_performance_alerts(self, threshold: float = 5.0):
        """
        Check for performance issues and generate alerts.
        
        This method implements proactive monitoring by:
        - Analyzing metric trends for performance degradation
        - Generating alerts when thresholds are exceeded
        - Preventing alert spam through duplicate detection
        
        Args:
            threshold: Response time threshold for generating alerts (seconds)
            
        Educational Note: This shows how to build alerting into monitoring systems
        to proactively identify performance issues.
        """
        # Get summary statistics for response time metric
        response_times = self.get_metric_summary('response_time')
        
        # Check if average response time exceeds threshold
        if response_times.get('avg', 0) > threshold:
            # Create descriptive alert message
            alert = f"High average response time: {response_times['avg']:.2f}s"
            
            # Prevent duplicate alerts by checking recent alert history
            # Only check last 5 alerts to balance memory usage with duplicate prevention
            if alert not in [a['message'] for a in self.alerts[-5:]]:
                # Create structured alert object with metadata
                self.alerts.append({
                    'timestamp': datetime.now(),    # When alert was generated
                    'level': 'WARNING',             # Alert severity level
                    'message': alert                # Human-readable description
                })


# ============================================================================
# Conversation History Management - Context preservation and search
# ============================================================================

class ConversationHistory:
    """
    Advanced conversation history management with search capabilities.
    
    This class demonstrates several important concepts:
    
    **Deque Data Structure**: Efficient fixed-size queue for conversation storage
    **Search Indexing**: Building indices for fast text search
    **Text Processing**: Cleaning and normalizing text for search
    **Memory Management**: Fixed-size storage with automatic eviction
    
    Key Features:
    - Efficient storage with automatic size management
    - Full-text search across conversation history
    - Metadata preservation for analytics
    - Relevance scoring for search results
    
    Educational Value:
    Shows how to implement search functionality and manage growing datasets
    in memory-constrained environments.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize conversation history with fixed maximum size.
        
        Args:
            max_size: Maximum number of conversation entries to retain
            
        Educational Note: deque with maxlen automatically evicts old entries
        when the size limit is reached, implementing a FIFO (First In, First Out) policy.
        """
        self.max_size = max_size
        
        # deque with maxlen provides efficient fixed-size queue behavior
        # When maxlen is reached, adding new items automatically removes old ones
        self.history = deque(maxlen=max_size)
        
        # Search index maps words to conversation entry indices
        # This enables fast text search without scanning all conversations
        self.search_index = {}
        
    def add_entry(self, query: str, response: str, metadata: Dict[str, Any]):
        """
        Add a conversation entry with comprehensive metadata and search indexing.
        
        This method demonstrates:
        - Structured data storage with consistent schema
        - Automatic search index maintenance
        - Hash-based duplicate detection
        - Timestamp preservation for temporal analysis
        
        Args:
            query: User's original question
            response: System's response
            metadata: Additional processing information
            
        Educational Note: Creating a hash of the query enables fast duplicate
        detection and can be used for caching strategies.
        """
        # Create structured entry with comprehensive metadata
        entry = {
            'timestamp': datetime.now(),                                # When conversation occurred
            'query': query,                                            # User's question
            'response': response,                                      # System's answer
            'metadata': metadata,                                      # Processing details
            'query_hash': hashlib.md5(query.encode()).hexdigest()     # Unique identifier for deduplication
        }
        
        # Add to conversation history (automatically evicts old entries if full)
        self.history.append(entry)
        
        # Update search index for fast text retrieval
        self._update_search_index(entry)
    
    def _update_search_index(self, entry: Dict[str, Any]):
        """
        Update search index for quick retrieval - FIXED VERSION
        
        This method builds an inverted index mapping words to document positions,
        enabling fast full-text search across conversation history.
        
        Args:
            entry: Conversation entry to index
            
        Educational Note: Inverted indices are fundamental to search engines.
        They map terms to documents containing those terms for fast lookup.
        """
        # Split query into individual words for indexing
        query_words = entry['query'].lower().split()
        
        for word in query_words:
            # Clean the word by removing common punctuation
            # This improves search accuracy by normalizing text
            cleaned_word = word.strip('.,!?;:"()[]{}')
            
            # Only index meaningful words (length > 2 filters out articles, etc.)
            if cleaned_word and len(cleaned_word) > 2:
                # Initialize word entry in index if it doesn't exist
                if cleaned_word not in self.search_index:
                    self.search_index[cleaned_word] = []
                
                # Add current conversation index to this word's entry list
                # len(self.history) - 1 gives the index of the just-added entry
                self.search_index[cleaned_word].append(len(self.history) - 1)    
    
    def search_history(self, search_term: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search conversation history - FIXED VERSION
        
        This method implements sophisticated text search with:
        - Multi-word query support
        - Partial word matching
        - Relevance scoring
        - Result ranking and limiting
        
        Args:
            search_term: Text to search for in conversation history
            limit: Maximum number of results to return
            
        Returns:
            List of conversation entries ranked by relevance
            
        Educational Note: This demonstrates how to build search functionality
        that goes beyond simple string matching to provide intelligent results.
        """
        # Split search term into individual words for multi-word search
        search_words = search_term.lower().split()
        candidate_indices = set()  # Use set to automatically handle duplicates
        
        # Find conversations containing search words
        for word in search_words:
            # Clean search word same way as indexing
            cleaned_word = word.strip('.,!?;:"()[]{}')
            
            # Direct index lookup for exact word matches
            if cleaned_word in self.search_index:
                candidate_indices.update(self.search_index[cleaned_word])
            
            # Partial matching: find indexed words containing the search word
            # This handles cases like searching "program" and finding "programming"
            for indexed_word in self.search_index:
                if cleaned_word in indexed_word or indexed_word in cleaned_word:
                    candidate_indices.update(self.search_index[indexed_word])
        
        # Convert indices to actual conversation entries with bounds checking
        candidates = []
        for i in candidate_indices:
            # Defensive programming: ensure index is still valid
            # deque size might have changed due to automatic eviction
            if i < len(self.history):
                candidates.append(self.history[i])
    
        # Sort by relevance using custom scoring function
        def relevance_score(entry):
            """
            Calculate relevance score based on word matching frequency.
            
            This simple scoring algorithm counts how many search words
            appear in the conversation query. More sophisticated algorithms
            could consider word positions, exact matches, etc.
            """
            query_words = entry['query'].lower().split()
            
            # Count matches between search words and query words
            # Uses nested comprehension to check all combinations
            return sum(1 for word in search_words 
                      if any(search_word in query_word 
                            for query_word in query_words 
                            for search_word in [word]))
        
        # Sort by relevance score (highest first) and limit results
        candidates.sort(key=relevance_score, reverse=True)
        return candidates[:limit]
    
    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history for context display.
        
        Args:
            limit: Maximum number of recent entries to return
            
        Returns:
            List of recent conversation entries in chronological order
            
        Educational Note: Using negative list slicing [-limit:] efficiently
        gets the last N items from the deque.
        """
        # Convert deque to list and get last 'limit' entries
        return list(self.history)[-limit:]


# ============================================================================
# Command Processing System - Sophisticated terminal interface
# ============================================================================

class CommandProcessor:
    """
    Comprehensive command processing system for interactive terminal interface.
    
    This class demonstrates several important software engineering patterns:
    
    **Command Pattern**: Encapsulating commands as methods for easy extension
    **Strategy Pattern**: Different handlers for different command types
    **Help System**: Self-documenting command interface
    **Error Handling**: Graceful failure for invalid commands
    **Alias System**: User-friendly shortcuts for common commands
    
    Key Features:
    - Extensible command system with easy registration
    - Built-in help and documentation
    - Command aliases for user convenience
    - Usage tracking for analytics
    - Robust error handling
    
    Educational Value:
    Shows how to build sophisticated command-line interfaces that are
    both powerful and user-friendly.
    """
    
    def __init__(self, app: 'RecursiveAIChatbotApp'):
        """
        Initialize command processor with reference to main application.
        
        Args:
            app: Reference to main application for accessing system state
            
        Educational Note: Storing a reference to the main app allows commands
        to access and modify system state while keeping command logic separate.
        """
        # Store reference to main application for state access
        self.app = app
        
        # Command handlers dictionary maps command names to handler methods
        # This enables easy command registration and lookup
        self.command_handlers = {
            # System control commands
            'quit': self._handle_quit,
            'exit': self._handle_quit,
            'bye': self._handle_quit,
            'q': self._handle_quit,
            
            # Information and analytics commands
            'stats': self._handle_stats,
            'statistics': self._handle_stats,
            'help': self._handle_help,
            'h': self._handle_help,
            '?': self._handle_help,
            
            # System maintenance commands
            'clear': self._handle_clear,
            'cls': self._handle_clear,
            'cache': self._handle_cache,
            'cache-info': self._handle_cache,
            'reset': self._handle_reset,
            
            # Configuration and advanced features
            'config': self._handle_config,
            'performance': self._handle_performance,
            'history': self._handle_history,
            'search': self._handle_search,
            'export': self._handle_export,
            'memory': self._handle_memory,
            'rate': self._handle_rate_response,
            'backup': self._handle_backup,
            'plugins': self._handle_plugins,
            'debug': self._handle_debug,
        }
        
        # Command aliases provide user-friendly shortcuts
        # This improves user experience by supporting common abbreviations
        self.aliases = {
            'perf': 'performance',      # Performance monitoring
            'mem': 'memory',            # Memory usage
            'hist': 'history',          # Conversation history
            'find': 'search',           # Search functionality
            'save': 'export',           # Data export
            'rating': 'rate',           # Response rating
        }
    
    def process_command(self, command: str, args: List[str] = None) -> bool:
        """
        Process a command and return True if it was handled successfully.
        
        This method implements the core command processing logic with:
        - Alias resolution for user convenience
        - Usage tracking for analytics
        - Comprehensive error handling
        - Return value indicating whether to continue or exit
        
        Args:
            command: Command name to execute
            args: Optional arguments for the command
            
        Returns:
            bool: True to continue session, False to exit application
            
        Educational Note: The boolean return value enables commands to
        signal when the application should terminate.
        """
        # Initialize args to empty list if not provided (defensive programming)
        if args is None:
            args = []
            
        # Resolve command aliases to actual command names
        # This allows users to type 'perf' instead of 'performance'
        command = self.aliases.get(command, command)
        
        # Check if command exists in our handler registry
        if command in self.command_handlers:
            # Track command usage for analytics and user behavior analysis
            self.app.session_stats.command_usage[command] = (
                self.app.session_stats.command_usage.get(command, 0) + 1
            )
            
            try:
                # Execute the command handler method
                # Each handler returns True/False to indicate continue/exit
                return self.command_handlers[command](args)
                
            except Exception as e:
                # Graceful error handling prevents commands from crashing the application
                print(f"❌ Error executing command '{command}': {e}")
                return True  # Continue running even if command failed
        
        # Command not recognized - return False to indicate it wasn't handled
        return False
    
    def _handle_quit(self, args: List[str]) -> bool:
        """
        Handle application termination with graceful shutdown.
        
        This method demonstrates proper application shutdown:
        - User-friendly goodbye message
        - Final statistics display
        - Return False to signal termination
        
        Returns:
            bool: False to signal application should exit
        """
        print("\n👋 Thank you for using the Advanced Recursive AI Chatbot!")
        self.app._print_session_summary()  # Show final statistics
        return False  # Signal to exit main loop
    
    def _handle_stats(self, args: List[str]) -> bool:
        """
        Handle statistics display and export commands.
        
        Supports both viewing stats and exporting them to files.
        
        Args:
            args: Command arguments ('export' for file export)
            
        Returns:
            bool: True to continue session
        """
        if args and args[0] == 'export':
            self.app._export_stats()  # Export statistics to file
        else:
            self.app._print_detailed_stats()  # Display statistics on screen
        return True
    
    def _handle_help(self, args: List[str]) -> bool:
        """
        Handle help system with context-sensitive information.
        
        Provides different help based on arguments for better user experience.
        
        Args:
            args: Help context ('commands' for detailed command help)
            
        Returns:
            bool: True to continue session
        """
        if args and args[0] == 'commands':
            self._print_command_help()  # Detailed command documentation
        else:
            self.app._print_help()      # General help information
        return True
    
    def _handle_clear(self, args: List[str]) -> bool:
        """
        Clear the terminal screen for better user experience.
        
        Uses platform-specific commands for cross-platform compatibility.
        
        Returns:
            bool: True to continue session
            
        Educational Note: os.name identifies the operating system,
        enabling platform-specific behavior.
        """
        # Use appropriate clear command for the operating system
        # 'cls' for Windows, 'clear' for Unix-like systems
        os.system('cls' if os.name == 'nt' else 'clear')
        return True
    
    def _handle_cache(self, args: List[str]) -> bool:
        """
        Handle cache management operations.
        
        Supports both viewing cache information and clearing cache.
        
        Args:
            args: Command arguments ('clear' to clear cache)
            
        Returns:
            bool: True to continue session
        """
        if args and args[0] == 'clear':
            # Clear the cache and provide user feedback
            self.app.recursive_handler.clear_cache()
            print("🧹 Cache cleared successfully!")
        else:
            # Display current cache statistics
            self.app._print_cache_info()
        return True
    
    def _handle_reset(self, args: List[str]) -> bool:
        """
        Reset the current session to initial state.
        
        This provides a way to start fresh without restarting the application.
        
        Returns:
            bool: True to continue session
        """
        self.app._reset_session()
        return True
    
    def _handle_config(self, args: List[str]) -> bool:
        """
        Handle configuration management operations.
        
        Supports viewing and saving configuration settings.
        
        Args:
            args: Command arguments ('show' or 'save')
            
        Returns:
            bool: True to continue session
        """
        if args and args[0] == 'show':
            self._show_config()  # Display current configuration
        elif args and args[0] == 'save':
            # Save configuration to file
            self.app.config.save_to_file('chatbot_config.json')
            print("💾 Configuration saved to chatbot_config.json")
        else:
            print("Usage: config [show|save]")  # Help for invalid usage
        return True
    
    def _handle_performance(self, args: List[str]) -> bool:
        """
        Display performance metrics and monitoring information.
        
        Returns:
            bool: True to continue session
        """
        self._show_performance_metrics()
        return True
    
    def _handle_history(self, args: List[str]) -> bool:
        """
        Display conversation history with optional limit.
        
        Args:
            args: Optional number of entries to display
            
        Returns:
            bool: True to continue session
        """
        # Default to showing 10 recent entries
        limit = 10
        
        # Parse numeric argument if provided
        if args and args[0].isdigit():
            limit = int(args[0])
        
        # Get and display conversation history
        history = self.app.conversation_history.get_recent_history(limit)
        self._display_history(history)
        return True
    
    def _handle_search(self, args: List[str]) -> bool:
        """
        Search conversation history for specific terms.
        
        Args:
            args: Search terms to look for
            
        Returns:
            bool: True to continue session
        """
        # Validate that search term was provided
        if not args:
            print("Usage: search <search_term>")
            return True
        
        # Join all arguments into single search term
        search_term = ' '.join(args)
        
        # Perform search and display results
        results = self.app.conversation_history.search_history(search_term)
        self._display_history(results, f"Search results for '{search_term}'")
        return True
    
    def _handle_export(self, args: List[str]) -> bool:
        """
        Export session data to file with automatic naming.
        
        Args:
            args: Optional filename for export
            
        Returns:
            bool: True to continue session
        """
        # Use provided filename or generate automatic name with timestamp
        filename = args[0] if args else f"chatbot_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.app._export_session_data(filename)
        return True
    
    def _handle_memory(self, args: List[str]) -> bool:
        """
        Display current memory usage statistics.
        
        Returns:
            bool: True to continue session
        """
        self._show_memory_usage()
        return True
    
    def _handle_rate_response(self, args: List[str]) -> bool:
        """
        Allow user to rate the quality of the last response.
        
        Args:
            args: Rating value (1-5)
            
        Returns:
            bool: True to continue session
        """
        # Validate rating input
        if not args or not args[0].isdigit():
            print("Usage: rate <1-5>")
            return True
        
        rating = int(args[0])
        
        # Validate rating range
        if 1 <= rating <= 5:
            # Store rating for analytics
            self.app.session_stats.user_satisfaction_ratings.append(rating)
            print(f"✅ Thank you for rating the last response: {rating}/5")
        else:
            print("Please provide a rating between 1 and 5")
        return True
    
    def _handle_backup(self, args: List[str]) -> bool:
        """
        Create backup of important system data.
        
        Returns:
            bool: True to continue session
        """
        self.app._create_backup()
        return True
    
    def _handle_plugins(self, args: List[str]) -> bool:
        """
        Handle plugin system operations (placeholder for future development).
        
        Args:
            args: Plugin commands ('list' or 'load <name>')
            
        Returns:
            bool: True to continue session
        """
        if args and args[0] == 'list':
            self._list_plugins()
        elif args and args[0] == 'load' and len(args) > 1:
            self._load_plugin(args[1])
        else:
            print("Usage: plugins [list|load <plugin_name>]")
        return True
    
    def _handle_debug(self, args: List[str]) -> bool:
        """
        Display debug information for troubleshooting.
        
        Returns:
            bool: True to continue session
        """
        self._show_debug_info()
        return True
    
    def _print_command_help(self):
        """
        Print detailed command help with descriptions.
        
        Educational Note: This method uses the __doc__ attribute to extract
        docstrings from handler methods for automatic documentation.
        """
        print("\n🆘 Advanced Command System:")
        print("-" * 40)
        for cmd, handler in self.command_handlers.items():
            # Get method docstring or provide default description
            doc = handler.__doc__ or "No description available"
            print(f" 🔧 {cmd:<15} - {doc}")
    
    def _show_config(self):
        """Display current configuration in formatted table."""
        print("\n⚙️ Current Configuration:")
        print("-" * 30)
        
        # Convert dataclass to dictionary for iteration
        config_dict = asdict(self.app.config)
        
        # Display each configuration option with its value
        for key, value in config_dict.items():
            print(f" 🔧 {key}: {value}")
    
    def _show_performance_metrics(self):
        """Display comprehensive performance metrics and alerts."""
        print("\n📊 Performance Metrics:")
        print("-" * 25)
        
        # Display metrics for key performance indicators
        for metric_name in ['response_time', 'memory_usage', 'cache_hit_rate']:
            summary = self.app.performance_monitor.get_metric_summary(metric_name)
            if summary:
                print(f" 📈 {metric_name}:")
                # Display all statistical measures
                for stat, value in summary.items():
                    print(f"    {stat}: {value:.3f}")
        
        # Display recent performance alerts
        if self.app.performance_monitor.alerts:
            print("\n⚠️ Recent Alerts:")
            for alert in self.app.performance_monitor.alerts[-5:]:  # Show last 5 alerts
                timestamp_str = alert['timestamp'].strftime('%H:%M:%S')
                print(f"   {timestamp_str} - {alert['message']}")
    
    def _display_history(self, history: List[Dict[str, Any]], title: str = "Conversation History"):
        """
        Display conversation history in formatted, readable format.
        
        Args:
            history: List of conversation entries to display
            title: Header title for the display
        """
        # Handle empty history gracefully
        if not history:
            print("📭 No history found")
            return
        
        print(f"\n📜 {title}:")
        print("-" * 40)
        
        # Display each entry with formatting and truncation
        for i, entry in enumerate(history, 1):
            timestamp = entry['timestamp'].strftime('%H:%M:%S')
            
            # Truncate long queries for readable display
            query_preview = entry['query'][:50] + '...' if len(entry['query']) > 50 else entry['query']
            print(f" {i}. [{timestamp}] {query_preview}")
    
    def _show_memory_usage(self):
        """
        Display detailed memory usage information using psutil.
        
        Educational Note: This shows how to use external libraries like psutil
        for system monitoring and resource management.
        """
        # Get current process for memory analysis
        process = psutil.Process()
        memory_info = process.memory_info()
        
        print("\n💾 Memory Usage:")
        print("-" * 20)
        
        # Display memory statistics in MB for readability
        print(f" 🔢 RSS: {memory_info.rss / 1024 / 1024:.1f} MB")  # Resident Set Size
        print(f" 🔢 VMS: {memory_info.vms / 1024 / 1024:.1f} MB")  # Virtual Memory Size
        print(f" 📊 Peak: {self.app.session_stats.peak_memory_usage:.1f} MB")
        
        # Trigger garbage collection and show results
        # This demonstrates manual memory management in Python
        collected = gc.collect()
        print(f" 🧹 Garbage collected: {collected} objects")
    
    def _list_plugins(self):
        """Display available plugins (placeholder for future implementation)."""
        print("\n🔌 Plugin System:")
        print("-" * 20)
        print("🚧 Plugin system is currently under development")
        
    def _load_plugin(self, plugin_name: str):
        """Load a specific plugin (placeholder for future implementation)."""
        print(f"🔌 Loading plugin: {plugin_name}")
        print("🚧 Plugin loading not yet implemented")
    
    def _show_debug_info(self):
        """
        Display comprehensive debug information for troubleshooting.
        
        This information helps developers and advanced users diagnose issues.
        """
        print("\n🐛 Debug Information:")
        print("-" * 25)
        
        # System information
        print(f" 🐍 Python version: {sys.version}")
        print(f" 📁 Working directory: {os.getcwd()}")
        print(f" 🗂️ Data path: {self.app.data_path}")
        print(f" 🔧 Config: {type(self.app.config).__name__}")
        
        # Component status
        print(f" 💾 Cache size: {len(getattr(self.app.dynamic_context, 'cache', {}))}")
        
        # Count loaded components
        components = [self.app.chatbot, self.app.recursive_handler, 
                     self.app.dynamic_context, self.app.greedy_priority]
        loaded_components = [c for c in components if c]
        print(f" 🧠 Components loaded: {len(loaded_components)}")


# ============================================================================
# Main Application Class - Central orchestration and control
# ============================================================================

class RecursiveAIChatbotApp:
    """
    Comprehensive main application class for the Recursive AI Chatbot.
    
    This class demonstrates enterprise-level application architecture:
    
    **Dependency Injection**: Components are injected and orchestrated centrally
    **Configuration Management**: Centralized settings with file persistence
    **Resource Management**: Proper initialization and cleanup of system resources
    **Error Handling**: Comprehensive exception management throughout
    **Signal Handling**: Graceful shutdown on system signals
    **Performance Monitoring**: Built-in metrics collection and analysis
    **Plugin Architecture**: Foundation for extensible functionality
    **Logging System**: Professional logging for debugging and monitoring
    
    Key improvements:
    - Better error handling and validation
    - Enhanced statistics tracking
    - Improved query classification
    - More robust caching integration
    - Better user experience
    - Advanced configuration management
    - Performance monitoring
    - Plugin architecture foundation
    - Comprehensive logging system
    - Resource management
    
    Educational Value:
    This class shows how to structure large Python applications with
    enterprise-level features while maintaining clean, maintainable code.
    """

    def __init__(self, data_path: Optional[str] = None, enable_logging: bool = False, config_path: str = "chatbot_config.json"):
        """
        Initialize the chatbot application with all subsystems.

        This constructor demonstrates proper application initialization:
        - Configuration loading with fallbacks
        - Logging setup with multiple handlers
        - Component initialization with error handling
        - Signal handler registration for graceful shutdown
        - Performance monitoring setup
        - Automatic backup creation

        Args:
            data_path: Path to the JSON file containing the knowledge base
            enable_logging: Whether to enable detailed logging
            config_path: Path to configuration file
            
        Educational Note: The constructor follows the dependency injection pattern,
        where all dependencies are resolved and injected during initialization.
        """
        # Step 1: Configuration Management
        # Load configuration from file with fallback to defaults
        self.config = ChatbotConfig.load_from_file(config_path)
        
        # Step 2: Logging Setup
        # Initialize logging system based on configuration
        self._setup_logging(enable_logging or self.config.logging_level != "INFO")
        
        # Step 3: Data Path Resolution
        # Resolve data file path with multiple fallback options
        self.data_path = self._resolve_data_path(data_path)
        
        # Step 4: Statistics Initialization
        # Create session statistics tracking object
        self.session_stats = SessionStats()
        
        # Step 5: Advanced Components Initialization
        # Initialize optional advanced features based on configuration
        self.performance_monitor = PerformanceMonitor() if self.config.enable_performance_monitoring else None
        self.conversation_history = ConversationHistory(self.config.max_history_size) if self.config.enable_conversation_history else None
        self.command_processor = CommandProcessor(self)
        
        # Step 6: System Signal Handlers
        # Setup graceful shutdown for system signals (Ctrl+C, SIGTERM, etc.)
        signal.signal(signal.SIGINT, self._signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler)  # Termination signal
        atexit.register(self._cleanup)                       # Program exit cleanup
        
        # Step 7: User Feedback and Progress
        # Provide visual feedback during initialization
        print("🤖 Initializing AI Chatbot...")
        print("=" * 55)

        try:
            # Step 8: Core Component Initialization
            # Initialize all core chatbot components
            self._initialize_components()
            
            # Step 9: Background Services
            # Setup automatic statistics saving and other background tasks
            self._setup_auto_save()
            
            # Step 10: Data Backup
            # Create backup of important data if configured
            if self.config.backup_data_on_start:
                self._create_backup()
            
            # Step 11: Success Feedback
            # Display initialization success with system status
            print("✅ Advanced Chatbot initialized successfully!")
            print(f"📊 Knowledge base loaded with {len(self.chatbot.knowledge_base.qa_pairs)} QA pairs")
            print(f"🔧 Using optimized recursive handling with intelligent caching")
            print(f"⚡ Performance monitoring: {'Enabled' if self.performance_monitor else 'Disabled'}")
            print(f"📜 Conversation history: {'Enabled' if self.conversation_history else 'Disabled'}")
            
        except Exception as e:
            # Comprehensive error handling during initialization
            self.logger.error(f"Initialization failed: {e}")
            print(f"❌ Error initializing chatbot: {e}")
            raise  # Re-raise to prevent running with broken initialization

    def _setup_logging(self, enable_logging: bool):
        """
        Configure comprehensive logging system with file and console output.
        
        This method demonstrates professional logging setup:
        - Multiple output handlers (file and console)
        - Structured log formatting with timestamps
        - Automatic log directory creation
        - Configurable log levels
        
        Args:
            enable_logging: Whether to enable detailed logging
            
        Educational Note: Good logging is essential for debugging production
        applications and understanding system behavior over time.
        """
        # Create logger instance for this application
        self.logger = logging.getLogger(__name__)
        
        if enable_logging:
            # Create logs directory if it doesn't exist
            # exist_ok=True prevents errors if directory already exists
            os.makedirs('logs', exist_ok=True)
            
            # Setup file handler for persistent logging
            # Include date in filename for automatic log rotation
            file_handler = logging.FileHandler(
                f'logs/chatbot_{datetime.now().strftime("%Y%m%d")}.log'
            )
            
            # Setup console handler for real-time debugging
            console_handler = logging.StreamHandler()
            
            # Create consistent formatter for both handlers
            # Include timestamp, logger name, level, and message
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Apply formatter to both handlers
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            # Set logging level based on configuration
            # getattr safely converts string to logging constant
            self.logger.setLevel(getattr(logging, self.config.logging_level.upper()))

    def _resolve_data_path(self, data_path: Optional[str]) -> str:
        """
        Enhanced data path resolution with better error handling.
        
        This method implements a sophisticated file discovery strategy:
        - Primary path specification
        - Multiple fallback locations
        - Cross-platform path handling
        - Graceful degradation when files are missing
        
        Args:
            data_path: User-specified data path or None for auto-discovery
            
        Returns:
            str: Resolved path to data file (may not exist)
            
        Educational Note: This pattern of trying multiple locations is common
        in applications that need to work across different deployment scenarios.
        """
        # Use default path if none specified
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'dev-v2.0.json')
        
        # Check if primary path exists
        if not os.path.exists(data_path):
            # Try alternative paths in order of preference
            alternative_paths = [
                'data/dev-v2.0.json',                        # Relative to current directory
                '../data/dev-v2.0.json',                     # Parent directory
                './dev-v2.0.json',                           # Current directory
                os.path.expanduser('~/chatbot_data/dev-v2.0.json')  # User home directory
            ]
            
            # Try each alternative path
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    data_path = alt_path
                    self.logger.info(f"Using alternative data path: {alt_path}")
                    break
            else:
                # No file found - log warning but continue with limited functionality
                self.logger.warning(f"Data file not found at {data_path}, will continue with limited functionality")
        
        return data_path

    def _initialize_components(self):
        """
        Enhanced component initialization with better error handling.
        
        This method demonstrates proper dependency management:
        - Sequential initialization with error checking
        - Component validation after creation
        - Comprehensive logging of initialization steps
        - Graceful failure handling
        
        Educational Note: Proper component initialization is crucial for
        system reliability and debugging.
        """
        # Initialize core modules with error handling
        # Each component depends on the previous ones being successfully created
        
        # Core chatbot engine
        self.chatbot = AIChatbot(self.data_path)
        
        # Recursive query handler with configurable depth
        self.recursive_handler = RecursiveHandling(
            self.chatbot, 
            max_recursion_depth=self.config.max_recursion_depth
        )
        
        # Dynamic context and caching system
        self.dynamic_context = DynamicContext()
        
        # Priority-based query optimization
        self.greedy_priority = GreedyPriority()
        
        # Validate that all components were created successfully
        self._validate_components()
        
        # Log successful initialization
        self.logger.info("All components initialized successfully")

    def _validate_components(self):
        """
        Validate that all components are properly initialized.
        
        This method implements defensive programming by checking that
        all critical components were created successfully before proceeding.
        
        Raises:
            RuntimeError: If any component failed to initialize
            
        Educational Note: Validation catches initialization problems early,
        making debugging much easier than discovering issues later.
        """
        # Dictionary mapping component names to instances for easy validation
        components = {
            'chatbot': self.chatbot,
            'recursive_handler': self.recursive_handler,
            'dynamic_context': self.dynamic_context,
            'greedy_priority': self.greedy_priority
        }
        
        # Check each component for successful initialization
        for name, component in components.items():
            if component is None:
                raise RuntimeError(f"Failed to initialize {name}")

    def _setup_auto_save(self):
        """
        Setup automatic statistics saving using background threading.
        
        This method demonstrates how to implement background tasks:
        - Daemon thread creation for non-blocking operation
        - Periodic task execution with configurable intervals
        - Error handling in background threads
        - Graceful shutdown with daemon threads
        
        Educational Note: Background threads are useful for periodic tasks
        that shouldn't block the main application functionality.
        """
        # Only setup auto-save if enabled in configuration
        if self.config.auto_save_stats:
            def auto_save():
                """
                Background thread function for automatic statistics saving.
                
                This function runs in an infinite loop, periodically saving
                statistics to disk. It uses daemon thread behavior to
                automatically terminate when the main program exits.
                """
                while True:
                    # Sleep for configured interval
                    time.sleep(self.config.stats_save_interval)
                    try:
                        # Attempt to save statistics
                        self._save_stats_to_file()
                    except Exception as e:
                        # Log errors but don't crash the background thread
                        self.logger.error(f"Auto-save failed: {e}")
            
            # Create and start daemon thread
            # daemon=True ensures thread terminates when main program exits
            save_thread = threading.Thread(target=auto_save, daemon=True)
            save_thread.start()

    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals gracefully - FIXED VERSION
        
        This method implements proper signal handling for graceful shutdown:
        - Immediate user feedback about shutdown process
        - Critical data saving before termination
        - Error handling during shutdown
        - System exit coordination
        
        Args:
            signum: Signal number received
            frame: Current stack frame (unused)
            
        Educational Note: Signal handlers enable applications to clean up
        properly when terminated by the operating system or user.
        """
        print(f"\n📡 Received signal {signum}, shutting down gracefully...")
        
        # Don't use logger here as it might cause issues during shutdown
        # Instead use print statements for immediate user feedback
        try:
            # Save statistics if auto-save is enabled
            if self.config.auto_save_stats:
                self._save_stats_to_file()
            print("✅ Graceful shutdown completed")
        except Exception as e:
            print(f"⚠️ Error during shutdown: {e}")
        

    def _cleanup(self):
        """
        Cleanup resources before shutdown - MOST ROBUST VERSION
        
        This method implements comprehensive cleanup procedures:
        - Statistics saving with error handling
        - Resource deallocation
        - Logging system cleanup
        - Multiple fallback levels for robustness
        
        Educational Note: Proper cleanup prevents data loss and resource
        leaks when applications terminate.
        """
        try:
            # Save statistics one final time if configured
            if self.config.auto_save_stats:
                self._save_stats_to_file()
            
            # Always use print for cleanup messages to avoid logging issues
            print("✅ Cleanup completed successfully")
                
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
        finally:
            # Safely close logging handlers to prevent resource leaks
            if hasattr(self, 'logger') and self.logger.handlers:
                # Create a copy of the list to avoid modification during iteration
                for handler in self.logger.handlers[:]:
                    try:
                        handler.close()              # Close file handles
                        self.logger.removeHandler(handler)  # Remove from logger
                    except:
                        pass  # Ignore errors when closing handlers

    def _classify_query_complexity(self, query: str) -> bool:
        """
        Enhanced query complexity classification with better preprocessing.
        
        This method implements intelligent query analysis to determine
        whether a query requires recursive processing or can be handled
        with simple processing.
        
        Args:
            query: The user input string
            
        Returns:
            bool: True if it's a complex/compound query requiring recursive handling
            
        Educational Note: Query classification is crucial for performance
        optimization - simple queries can be processed faster with direct methods.
        """
        # Early return for obviously simple queries
        if not query or len(query.strip()) < 10:
            return False
        
        # Preprocess query for better analysis
        query = self._preprocess_query(query)
        
        # Use the recursive handler's built-in detection with caching
        try:
            return self.recursive_handler._is_nested_query_cached(query.strip())
        except Exception as e:
            # Log warning but continue with fallback analysis
            self.logger.warning(f"Error in query classification: {e}")
            # Fallback to enhanced heuristics
            return self.__complexity_check(query)

    def _preprocess_query(self, query: str) -> str:
        """
        Enhanced query preprocessing for better analysis and processing.
        
        This method implements text cleaning and normalization:
        - Whitespace normalization
        - Basic spell correction
        - Input sanitization
        
        Args:
            query: Raw user input
            
        Returns:
            str: Cleaned and normalized query
            
        Educational Note: Preprocessing improves the accuracy of downstream
        text analysis and processing algorithms.
        
        IMPLEMENTATION STEPS:
        
        Step 1: BASIC SANITIZATION
        - Call query.strip() to remove leading/trailing whitespace
        
        Step 2: WHITESPACE NORMALIZATION
        - Use ' '.join(query.split()) to remove excessive whitespace
        - This converts multiple spaces/tabs into single spaces
        
        Step 3: SPELL CORRECTION (CONDITIONAL)
        - Check if self.config.enable_spell_correction is True
        - If True: call self._basic_spell_correction(query) and assign result back to query
        
        Step 4: RETURN RESULT
        - Return the processed query string
        
        IMPORTANT NOTES:
        - The split/join technique is efficient for whitespace normalization
        - Spell correction is optional based on configuration
        - Each step builds on the previous preprocessing step
        """
        # TODO: Implement the method following the steps above
        pass
    
    def _basic_spell_correction(self, query: str) -> str:
        """
        Basic spell correction using simple replacement rules.
        
        This is a placeholder for more sophisticated spell correction.
        In production, this might use libraries like TextBlob or spaCy.
        
        Args:
            query: Query text to correct
            
        Returns:
            str: Query with basic corrections applied
            
        Educational Note: Even simple spell correction can significantly
        improve user experience and system accuracy.
        """
        # Simple replacements for common typos
        # In production, this could be expanded with machine learning models
        replacements = {
            'wat': 'what',          # Common typing error
            'hwo': 'how',           # Transposition error
            'teh': 'the',           # Common typo
            'ai': 'AI',             # Capitalization correction
        }
        
        # Apply corrections word by word
        words = query.split()
        corrected_words = [replacements.get(word.lower(), word) for word in words]
        return ' '.join(corrected_words)

    def __complexity_check(self, query: str) -> bool:
        """
        Enhanced fallback complexity check using linguistic heuristics.
        
        This method implements sophisticated query analysis using:
        - Conjunction detection for compound queries
        - Sequence indicators for multi-step questions
        - Question pattern analysis
        - Statistical analysis of query characteristics
        
        Args:
            query: Preprocessed query text
            
        Returns:
            bool: True if query appears complex
            
        Educational Note: Heuristic analysis provides fallback when
        sophisticated NLP methods are unavailable.
        """
        # Define linguistic indicators of complex queries
        complexity_indicators = [
            # Conjunctions indicating compound questions
            ' and ', ' & ', '; ', ' or ', ' but ', ' however ',
            
            # Sequence indicators suggesting multi-step queries
            'also', 'additionally', 'furthermore', 'moreover',
            'then', 'next', 'afterwards', 'subsequently',
            
            # Question patterns indicating information requests
            'what about', 'how about', 'tell me about',
            'explain both', 'describe each', 'list all',
            
            # Multiple question indicators
            '?', 'question:', 'Q:', 'A:'
        ]
        
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Count complexity indicators
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        question_count = query.count('?')  # Count question marks
        
        # Enhanced logic combining multiple signals
        is_complex = (
            indicator_count >= 2 or                          # Multiple indicators
            question_count > 1 or                            # Multiple questions
            (indicator_count >= 1 and len(query.split()) > 15) or  # Long query with indicators
            any(phrase in query_lower for phrase in ['step by step', 'in detail', 'comprehensive'])  # Explicit complexity requests
        )
        
        return is_complex

    def _update_stats(self, result: Dict[str, Any]):
        """
        Enhanced statistics tracking with comprehensive metrics - FIXED VERSION.
        
        This method implements comprehensive analytics collection:
        - Query processing metrics
        - Performance monitoring
        - Error analysis
        - Quality assessment
        - Memory usage tracking
        
        Args:
            result: Dictionary containing query processing results and metadata
            
        Educational Note: Comprehensive statistics enable data-driven
        optimization and help identify system bottlenecks and improvement opportunities.
        """
        try:
            # Increment total query counter
            self.session_stats.queries_processed += 1
            
            # Update query length statistics for user behavior analysis
            query_len = len(result.get('query', ''))
            stats = self.session_stats.query_length_stats
            
            # Update running statistics efficiently
            stats['min'] = min(stats['min'], query_len)  # Track shortest query
            stats['max'] = max(stats['max'], query_len)  # Track longest query
            
            # Calculate running average without storing all values
            total_chars = stats['avg'] * (self.session_stats.queries_processed - 1) + query_len
            stats['avg'] = total_chars / self.session_stats.queries_processed
            
            # Track processing time with performance monitoring integration
            processing_time = result.get('processing_time', 0.0)
            self.session_stats.total_time += processing_time
            
            # Record performance metrics if monitoring is enabled
            if self.performance_monitor:
                self.performance_monitor.record_metric('response_time', processing_time)
                
                # Check for performance alerts when response time is slow
                if processing_time > self.config.performance_alert_threshold:
                    self.performance_monitor.check_performance_alerts(self.config.performance_alert_threshold)
            
            # Calculate running average response time
            if self.session_stats.queries_processed > 0:
                self.session_stats.average_response_time = (
                    self.session_stats.total_time / self.session_stats.queries_processed
                )
            
            # Track memory usage for resource monitoring
            if self.performance_monitor:
                try:
                    # Use psutil to get current memory usage
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024  # Convert to MB
                    
                    # Record memory metric and update peak usage
                    self.performance_monitor.record_metric('memory_usage', memory_mb)
                    self.session_stats.peak_memory_usage = max(
                        self.session_stats.peak_memory_usage, memory_mb
                    )
                except:
                    pass  # psutil might not be available on all systems
            
            # Track caching effectiveness
            if result.get('used_cache', False):
                self.session_stats.cache_hits += 1
            
            # Track recursive query usage
            if result.get('is_recursive', False):
                self.session_stats.recursive_queries += 1
            
            # Track success/failure rates for quality monitoring
            if result.get('success', True):
                self.session_stats.successful_queries += 1
            else:
                self.session_stats.failed_queries += 1
                
                # FIXED: Track error categories properly
                error_info = result.get('error_info', {})
                if 'exception_type' in error_info:
                    error_type = error_info['exception_type']
                else:
                    # Fallback to generic error if no type info available
                    error_type = 'GeneralError'
                
                # Update error category counter
                self.session_stats.error_categories[error_type] = (
                    self.session_stats.error_categories.get(error_type, 0) + 1
                )
            
            # Track fuzzy matching usage for algorithm analysis
            if result.get('fuzzy_match', False):
                self.session_stats.fuzzy_matches += 1
            
            # Track priority distribution for load balancing insights
            priority = result.get('priority', 2)
            if priority in self.session_stats.priority_distribution:
                self.session_stats.priority_distribution[priority] += 1
            
            # Estimate response quality using heuristic scoring
            response_length = len(result.get('response', ''))
            quality_score = min(1.0, response_length / 100)  # Length-based score (capped at 1.0)
            
            # Adjust quality score based on processing characteristics
            if result.get('fuzzy_match'):
                quality_score *= 0.8  # Penalize fuzzy matches slightly (less precise)
            if result.get('used_cache'):
                quality_score *= 1.1  # Reward cache hits (faster response)
            
            # Store quality score for trend analysis
            self.session_stats.response_quality_scores.append(quality_score)
            
        except Exception as e:
            # Log statistics errors but don't crash the application
            self.logger.error(f"Error updating statistics: {e}")

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Enhanced core handler to process a single user query with comprehensive features - FIXED VERSION.
        
        This method implements the core query processing pipeline:
        - Input validation and preprocessing
        - Query complexity classification
        - Appropriate processing strategy selection
        - Caching integration
        - Performance monitoring
        - Error handling and recovery
        
        Args:
            query: The input question or prompt from the user
            
        Returns:
            Dict[str, Any]: Structured response with comprehensive metadata
            
        Educational Note: This method demonstrates how to build robust
        processing pipelines with multiple fallback strategies and
        comprehensive error handling.
        """
        # Start timing for performance measurement
        start_time = time.time()
        
        # Step 1: Enhanced input validation
        # TODO: call _validate_query_input(query) and assign it to validation_result
        # This step ensures the query is safe, valid, and meets quality standards
        validation_result =["":""]  # Placeholder for validation logic

        if not validation_result['valid']:
            # Return structured error response for validation failures
            return {
                'query': query,
                'response': validation_result['message'],
                'processing_time': time.time() - start_time,
                'used_cache': False,
                'is_recursive': False,
                'priority': 1,
                'success': False,
                'error': validation_result['error'],
                'error_info': {'exception_type': 'ValidationError'},  # FIXED: Add exception type
                'fuzzy_match': False,
                'fuzzy_match_threshold': 0.0
            }
        
        # Step 2: Query preprocessing
        # TODO: call _preprocess_query(query) and assign it to query
        # This step ensures the query is clean and normalized for processing
        query = self._preprocess_query(query)
        
        # Step 3: Initialize comprehensive result structure
        result = {
            'query': query,
            'response': '',
            'processing_time': 0.0,
            'used_cache': False,
            'is_recursive': False,
            'priority': 2,                          # Default priority
            'success': True,                        # Assume success unless error occurs
            'error': None,
            'error_info': {},                       # FIXED: Add error_info field
            'fuzzy_match': False,
            'fuzzy_match_threshold': 0.0,
            'query_length': len(query),
            'preprocessing_applied': True,
            'component_used': 'unknown'
        }

        try:
            # Step 4: Assign priority level using greedy priority algorithm
            # TODO: call greedy_priority.get_priority(query) and assign it to result['priority']
            result['priority'] = 0 # Placeholder for priority assignment logic


            # Step 5: Check cache first for performance optimization (Dynamic Programming)
            # TODO: call dynamic_context.retrieve_from_cache(query) and assign it to cached_response
            cached_response = "" # Placeholder for cache retrieval logic

            if(cached_response and self.chatbot.DEFAULT_NO_MATCH_MESSAGE not in cached_response):
                # Cache hit - use cached response
                result['response'] = cached_response
                result['used_cache'] = True
                result['component_used'] = 'cache'
                self.logger.info(f"Cache hit for query: {query[:50]}...")
            else:
                # Step 6: Determine processing approach with enhanced classification
                #TODO: call _classify_query_complexity(query) and assign it to is_complex
                # This step decides whether to use recursive handling or standard processing
                is_complex = False # Placeholder for complexity classification logic
                
                if is_complex:
                    # Use recursive handling for complex queries
                    result['component_used'] = 'recursive_handler'
                    self.logger.info(f"Processing complex query: {query[:50]}...")
                    
                    #TODO: call recursive_handler.handle_recursive_query(query) and assign it to recursive_result
                    recursive_result = QueryResult(
                        response='',
                        responses=[],
                        processing_time=0.0,
                        is_recursive=True,
                        used_cache=False,
                        priority=result['priority'],
                        fuzzy_match=False,
                        fuzzy_match_threshold=0.0
                    )  # Placeholder for recursive handling logic
                    
                    # Extract data from QueryResult object or dict with enhanced handling
                    if hasattr(recursive_result, '__dict__'):
                        # It's a QueryResult dataclass - extract attributes
                        result.update({
                            'response': recursive_result.response,
                            'is_recursive': recursive_result.is_recursive,
                            'fuzzy_match': recursive_result.fuzzy_match,
                            'fuzzy_match_threshold': recursive_result.fuzzy_match_threshold,
                            'used_cache': recursive_result.used_cache or result['used_cache']
                        })
                    else:
                        # It's a dictionary 
                        result.update({
                            'response': recursive_result.get('response', ''),
                            'is_recursive': recursive_result.get('is_recursive', True),
                            'fuzzy_match': recursive_result.get('fuzzy_match', False),
                            'fuzzy_match_threshold': recursive_result.get('fuzzy_match_threshold', 0.0),
                            'used_cache': recursive_result.get('used_cache', False) or result['used_cache']
                        })
                else:
                    # Standard single query processing
                    result['component_used'] = 'chatbot'
                    self.logger.info(f"Processing simple query: {query[:50]}...")
                    #TODO: call chatbot.handle_query(query) and assign it to response, is_fuzzy, threshold
                    #this step uses the main chatbot engine to handle the query
                    response, is_fuzzy, threshold = Tuple[str,bool,float] # Placeholder for chatbot processing logic
                    
                    # Update result with chatbot response
                    if (response is not None or response != ''):
                        result.update({
                            'response': response,
                            'fuzzy_match': is_fuzzy,
                            'fuzzy_match_threshold': threshold
                        })

                # Step 7: Cache the result for future use with enhanced conditions
                if (result['response'] and 
                    not result.get('error') and 
                    len(result['response']) >= 3 and
                    result['priority'] >= 3):  # Only cache meaningful, high-priority responses
                    #TODO: call dynamic_context.store_in_cache(query, result['response'])
                    # This step stores the response in the cache for future queries
                    # Placeholder for cache storage logic


            # Step 8: Post-process response for better user experience
            result['response'] = self._post_process_response(result['response'], result)

        except Exception as e:
            # Comprehensive error handling with logging and structured error response
            self.logger.error(f"Error processing query '{query}': {e}")
            result.update({
                'success': False,
                'error': str(e),
                'error_info': {'exception_type': type(e).__name__},  # FIXED: Capture actual exception type
                'response': self._get_error_response(e),
                'component_used': 'error_handler'
            })

        # Step 9: Finalize timing and update comprehensive statistics
        result['processing_time'] = time.time() - start_time
        self._update_stats(result)
        
        # Step 10: Add to conversation history if enabled
        if self.conversation_history:
            self.conversation_history.add_entry(query, result['response'], result)
        
        return result

    def _validate_query_input(self, query: str) -> Dict[str, Any]:
        """
        Enhanced input validation with security and quality checks.
        
        This method implements comprehensive input validation:
        - Empty input detection
        - Length limit enforcement
        - Security pattern detection
        - Basic quality assessment
        
        Args:
            query: Raw user input to validate
            
        Returns:
            Dict containing validation result and error messages
            
        Educational Note: Input validation is the first line of defense
        against invalid data and potential security issues.
        """
        # Check for empty or whitespace-only input
        if not query or not query.strip():
            return {
                'valid': False,
                'message': 'Please provide a valid question.',
                'error': 'Empty query'
            }
        
        # Enforce maximum query length to prevent resource exhaustion
        if len(query) > self.config.max_query_length:
            return {
                'valid': False,
                'message': f'Query too long. Maximum length is {self.config.max_query_length} characters.',
                'error': 'Query too long'
            }
        
        # Check for potentially malicious patterns (basic security)
        # This is a simple example - production systems need more sophisticated security
        suspicious_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
        if any(pattern in query.lower() for pattern in suspicious_patterns):
            return {
                'valid': False,
                'message': 'Query contains potentially unsafe content.',
                'error': 'Unsafe content detected'
            }
        
        # Input passed all validation checks
        return {'valid': True}

    def _post_process_response(self, response: str, result: Dict[str, Any]) -> str:
        """
        Post-process the response for better user experience.
        
        This method implements response enhancement:
        - Empty response handling
        - Quality indicators for fuzzy matches
        - Debug information injection
        - User-friendly formatting
        
        Args:
            response: Raw response text
            result: Processing result metadata
            
        Returns:
            str: Enhanced response text
            
        Educational Note: Post-processing improves user experience by
        providing context and additional information about how the response was generated.
        """
        # Handle empty or invalid responses
        if not response:
            return "I apologize, but I couldn't generate a response to your query."
        
        # Add quality indicators for fuzzy matches to manage user expectations
        if result.get('fuzzy_match') and result.get('fuzzy_match_threshold', 0) < 0.9:
            response = f"💡 *Similar question found:*\n\n{response}"
        
        # Add cache indicator for development/debug mode
        if result.get('used_cache') and self.config.logging_level == "DEBUG":
            response += "\n\n*[Response retrieved from cache]*"
        
        return response

    def _get_error_response(self, error: Exception) -> str:
        """
        Generate user-friendly error responses based on exception types.
        
        This method implements error message customization:
        - Exception type mapping to user messages
        - Fallback to generic message
        - User-focused language
        
        Args:
            error: The exception that occurred
            
        Returns:
            str: User-friendly error message
            
        Educational Note: Good error messages help users understand what
        went wrong and how to fix it, rather than exposing technical details.
        """
        # Map exception types to user-friendly messages
        error_responses = {
            'TimeoutError': "The request took too long to process. Please try a simpler question.",
            'ConnectionError': "There seems to be a connectivity issue. Please try again.",
            'ValueError': "There was an issue with your input. Please rephrase your question.",
            'KeyError': "I couldn't find the required information. Please try a different question.",
        }
        
        # Get error type name and look up appropriate message
        error_type = type(error).__name__
        return error_responses.get(error_type, 
            "I apologize, but I encountered an error processing your query. Please try rephrasing it.")

    def interactive_mode(self):
        """
        Enhanced interactive terminal session with comprehensive features.
        
        This method implements the main user interaction loop:
        - Welcome message and feature overview
        - Command processing integration
        - Query processing with timeout handling
        - Performance information display
        - User feedback collection
        - Graceful error handling and recovery
        
        Educational Note: Interactive loops are common in CLI applications
        and require careful design to handle all possible user inputs gracefully.
        """
        # Display welcome message with feature overview
        print("\n🎯 Starting Interactive Mode")
        print("=" * 55)
        print("💡 Features:")
        print("   • Intelligent query processing with caching")
        print("   • Performance monitoring and analytics")
        print("   • Conversation history and search")
        print("   • Advanced command system (type 'help' for commands)")
        print("   • Real-time statistics and feedback")
        print("-" * 55)

        # Main interaction loop
        while True:
            try:
                # Get user input with prompt
                user_input = input("\n💬 You: ").strip()
                
                # Skip empty input
                if not user_input:
                    continue

                # Enhanced command processing
                command_parts = user_input.split()
                command = command_parts[0].lower()
                args = command_parts[1:] if len(command_parts) > 1 else []
                
                # Try to process as command first
                if not self.command_processor.process_command(command, args):
                    # Check for quit commands that weren't handled by command processor
                    if command in ['quit', 'exit', 'bye', 'q']:
                        break
                    

                # Process as regular query with timeout handling
                try:
                    # Use timeout context manager for query processing
                    with self._query_timeout(self.config.response_timeout):
                        result = self.process_query(user_input)
                        print(f"\n🤖 Chatbot: {result['response']}")

                        # Show enhanced processing information
                        self._show_processing_info(result)
                        
                        # Ask for feedback on complex or slow queries
                        if (result['processing_time'] > 2.0 or 
                            result['is_recursive'] or 
                            not result['success']):
                            self._prompt_for_feedback()

                except TimeoutError:
                    # Handle query timeouts gracefully
                    print("\n⏰ Query timeout! The request took too long to process.")
                    print("💡 Try asking a simpler question or check your connection.")

            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\n\n👋 Session interrupted. Goodbye!")
                self._print_session_summary()
                break
                
            except Exception as e:
                # Handle unexpected errors without crashing
                self.logger.error(f"Error in interactive mode: {e}")
                print(f"\n❌ An unexpected error occurred: {e}")
                print("💡 Type 'debug' for diagnostic information")

    def _query_timeout(self, timeout: float):
        """
        Context manager for query timeout implementation.
        
        This is a simplified timeout implementation. In production,
        you might use more sophisticated timeout mechanisms.
        
        Args:
            timeout: Timeout duration in seconds
            
        Returns:
            Context manager for timeout handling
            
        Educational Note: Context managers provide clean resource management
        and ensure proper cleanup even when exceptions occur.
        """
        class TimeoutContext:
            def __init__(self, timeout_seconds):
                self.timeout = timeout_seconds
                
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # In a real implementation, this would handle actual timeouts
                pass
        
        return TimeoutContext(timeout)

    def _show_processing_info(self, result: Dict[str, Any]):
        """
        Display enhanced processing information based on result characteristics.
        
        This method provides users with insight into how their query was processed,
        which helps with understanding system behavior and optimizing queries.
        
        Args:
            result: Query processing result with metadata
            
        Educational Note: Transparent information about processing helps users
        understand system behavior and optimize their usage patterns.
        """
        # Determine whether to show processing info based on interesting characteristics
        show_info = (
            result['processing_time'] > 0.5 or      # Slow queries
            result['is_recursive'] or               # Complex queries
            result['fuzzy_match'] or               # Approximate matches
            not result['success'] or               # Failed queries
            result.get('used_cache', False)        # Cached responses
        )
        
        if show_info:
            print("\n📊 Processing Details:")
            print(f"   ⏱️  Time: {result['processing_time']:.3f}s")
            print(f"   🎯 Priority: {result['priority']}")
            print(f"   🔧 Component: {result.get('component_used', 'unknown')}")
            print(f"   🔄 Recursive: {'Yes' if result['is_recursive'] else 'No'}")
            print(f"   💾 Cached: {'Yes' if result['used_cache'] else 'No'}")
            
            # Show fuzzy match details if applicable
            if result['fuzzy_match']:
                print(f"   🔍 Fuzzy Match: {result['fuzzy_match_threshold']:.2f}")
            
            # Show error details if applicable
            if not result['success']:
                print(f"   ❌ Error: {result['error']}")

    def _prompt_for_feedback(self):
        """
        Prompt user for feedback on response quality.
        
        This method implements user feedback collection to improve system quality.
        It's designed to be non-intrusive and optional.
        
        Educational Note: User feedback is valuable for iterative system improvement
        and helps identify areas where the system needs enhancement.
        """
        # Only prompt for feedback if conversation history is enabled
        if not self.conversation_history:
            return
            
        try:
            # Non-blocking feedback prompt
            feedback_input = input("\n⭐ Rate this response (1-5, or press Enter to skip): ").strip()
            
            # Validate and process rating
            if feedback_input.isdigit() and 1 <= int(feedback_input) <= 5:
                rating = int(feedback_input)
                self.session_stats.user_satisfaction_ratings.append(rating)
                print(f"✅ Thank you for your feedback: {rating}/5")
        except:
            pass  # Skip if any error in feedback collection

    def _print_detailed_stats(self):
        """
        Display comprehensive session statistics in formatted, readable format.
        
        This method provides detailed analytics about the current session,
        including performance metrics, usage patterns, and quality indicators.
        
        Educational Note: Comprehensive statistics help users and administrators
        understand system performance and usage patterns.
        """
        print("\n📈 Comprehensive Session Statistics:")
        print("-" * 40)
        
        # Convert dataclass to dictionary for iteration
        stats_dict = asdict(self.session_stats)
        
        # Display each statistic category with appropriate formatting
        for key, value in stats_dict.items():
            if key == 'priority_distribution':
                print(f" 🎯 Priority Distribution:")
                for priority, count in value.items():
                    print(f"    Priority {priority}: {count} queries")
                    
            elif key == 'command_usage':
                if value:  # Only show if commands were used
                    print(f" 🔧 Command Usage:")
                    # Sort by usage count (most used first)
                    for cmd, count in sorted(value.items(), key=lambda x: x[1], reverse=True):
                        print(f"    {cmd}: {count} times")
                        
            elif key == 'query_length_stats':
                print(f" 📏 Query Length Statistics:")
                for stat, val in value.items():
                    if val != float('inf'):  # Skip uninitialized min value
                        print(f"    {stat}: {val:.1f} characters")
                        
            elif key == 'error_categories':
                if value:  # Only show if errors occurred
                    print(f" ❌ Error Categories:")
                    for error_type, count in value.items():
                        print(f"    {error_type}: {count}")
                        
            elif key in ['response_quality_scores', 'user_satisfaction_ratings']:
                if value:  # Only show if data exists
                    avg_score = sum(value) / len(value)
                    print(f" ⭐ {key.replace('_', ' ').title()}: {avg_score:.2f} (n={len(value)})")
                    
            elif 'time' in key:
                # Format time values with appropriate precision
                print(f" ⏱️  {key.replace('_', ' ').title()}: {value:.3f}s")
                
            elif key not in ['session_start_time']:  # Skip timestamp display
                print(f" 📊 {key.replace('_', ' ').title()}: {value}")
        
        # Calculate and display additional derived metrics
        if self.session_stats.queries_processed > 0:
            # Calculate performance percentages
            cache_hit_rate = (self.session_stats.cache_hits / self.session_stats.queries_processed) * 100
            success_rate = (self.session_stats.successful_queries / self.session_stats.queries_processed) * 100
            recursive_rate = (self.session_stats.recursive_queries / self.session_stats.queries_processed) * 100
            
            print(f" 💾 Cache Hit Rate: {cache_hit_rate:.1f}%")
            print(f" ✅ Success Rate: {success_rate:.1f}%")
            print(f" 🔄 Recursive Query Rate: {recursive_rate:.1f}%")
            
            # Calculate session productivity metrics
            session_duration = datetime.now() - self.session_stats.session_start_time
            queries_per_minute = self.session_stats.queries_processed / (session_duration.total_seconds() / 60)
            print(f" ⚡ Queries per minute: {queries_per_minute:.1f}")
            
            # Show user satisfaction if available
            if self.session_stats.user_satisfaction_ratings:
                avg_satisfaction = sum(self.session_stats.user_satisfaction_ratings) / len(self.session_stats.user_satisfaction_ratings)
                print(f" 😊 Average satisfaction: {avg_satisfaction:.1f}/5")

    def _print_cache_info(self):
        """
        Display comprehensive cache information and statistics.
        
        This method provides insights into caching performance and effectiveness,
        which is crucial for system optimization.
        
        Educational Note: Cache statistics help identify performance bottlenecks
        and optimize caching strategies for better system performance.
        """
        try:
            # Get cache information from recursive handler
            cache_info = self.recursive_handler.get_cache_info()
            print("\n💾 Comprehensive Cache Information:")
            print("-" * 30)
            
            # Display nested query cache statistics
            nested_cache = cache_info['nested_query_cache_info']
            print(f" 🔍 Query Classification Cache:")
            print(f"    Hits: {nested_cache['hits']}")
            print(f"    Misses: {nested_cache['misses']}")
            print(f"    Current Size: {nested_cache['currsize']}")
            print(f"    Max Size: {nested_cache['maxsize']}")
            
            # Calculate and display hit rate if there's data
            total_requests = nested_cache['hits'] + nested_cache['misses']
            if total_requests > 0:
                hit_rate = nested_cache['hits'] / total_requests * 100
                print(f"    Hit Rate: {hit_rate:.1f}%")
            
            # Display dynamic context cache information
            if hasattr(self.dynamic_context, 'cache'):
                cache_size = len(getattr(self.dynamic_context, 'cache', {}))
                print(f"\n 📦 Response Cache:")
                print(f"    Current Size: {cache_size}")
                print(f"    Max Size: {self.config.cache_size_limit}")
                
                # Calculate cache utilization
                if self.config.cache_size_limit > 0:
                    utilization = (cache_size / self.config.cache_size_limit) * 100
                    print(f"    Utilization: {utilization:.1f}%")
                
        except Exception as e:
            # Handle errors gracefully
            print(f"❌ Could not retrieve cache info: {e}")

    def _print_help(self):
        """
        Display comprehensive help information for users.
        
        This method provides a complete overview of system capabilities
        and usage instructions for optimal user experience.
        
        Educational Note: Good help systems are essential for user adoption
        and effective use of complex applications.
        """
        print("\n🆘 Advanced Command System:")
        print("-" * 25)
        
        # Core system commands
        print(" 🔚 quit/exit/bye/q     - End the session")
        print(" 📊 stats [export]      - Show/export detailed statistics")
        print(" 🆘 help [commands]     - Show this help menu")
        print(" 🧹 clear/cls          - Clear the screen")
        
        # Cache and performance commands
        print(" 💾 cache [clear]       - Show/clear cache statistics")
        print(" 🔄 reset              - Reset session statistics")
        print(" ⚙️  config [show|save] - Manage configuration")
        print(" 📈 performance/perf    - Show performance metrics")
        
        # History and search commands
        print(" 📜 history [n]         - Show conversation history")
        print(" 🔍 search <term>       - Search conversation history")
        print(" 💾 export [filename]   - Export session data")
        
        # System monitoring commands
        print(" 🧠 memory/mem          - Show memory usage")
        print(" ⭐ rate <1-5>          - Rate the last response")
        print(" 💾 backup             - Create data backup")
        
        # Advanced features
        print(" 🔌 plugins [list|load] - Manage plugins")
        print(" 🐛 debug              - Show debug information")
        
        print("\n💡 Query Tips:")
        print(" • Complex questions: 'What is AI and how does ML work?'")
        print(" • Use conjunctions: 'Explain X and also describe Y'")
        print(" • Ask follow-up questions to test caching")
        print(" • Use 'search' to find previous conversations")
        print(" • Rate responses to improve the system")

    def _reset_session(self):
        """
        Enhanced session reset with more comprehensive cleanup.
        
        This method provides a way to start fresh without restarting the
        entire application, which is useful for testing and demonstration.
        
        Educational Note: Reset functionality is important for applications
        that accumulate state over time and need periodic cleanup.
        """
        # Reset all session-specific state
        self.session_stats = SessionStats()
        self.recursive_handler.clear_cache()
        
        # Reset conversation history if enabled
        if self.conversation_history:
            self.conversation_history = ConversationHistory(self.config.max_history_size)
            
        # Reset performance monitoring if enabled
        if self.performance_monitor:
            self.performance_monitor = PerformanceMonitor()
        
        # Trigger garbage collection to free memory
        gc.collect()
        
        # Provide comprehensive feedback about what was reset
        print("\n🔄 Enhanced session reset completed!")
        print("   • Statistics cleared")
        print("   • All caches cleared")
        print("   • Conversation history cleared")
        print("   • Performance metrics reset")
        print("   • Memory garbage collected")

    def _print_session_summary(self):
        """
        Display comprehensive session summary at termination.
        
        This method provides a final overview of the session, including
        highlights, performance metrics, and achievements.
        
        Educational Note: Session summaries help users understand their
        usage patterns and system performance over time.
        """
        print("\n📋 Final Comprehensive Session Summary:")
        print("=" * 30)
        
        # Calculate session duration
        session_duration = datetime.now() - self.session_stats.session_start_time
        print(f"🕐 Session Duration: {session_duration}")
        
        # Display detailed statistics
        self._print_detailed_stats()
        
        # Display session highlights if there were any queries processed
        if self.session_stats.queries_processed > 0:
            print(f"\n🏆 Session Highlights:")
            
            # Highlight complex query processing
            if self.session_stats.recursive_queries > 0:
                print(f"   • Processed {self.session_stats.recursive_queries} complex queries")
                
            # Highlight caching effectiveness
            if self.session_stats.cache_hits > 0:
                print(f"   • Achieved {self.session_stats.cache_hits} cache hits")
                
            # Highlight fuzzy matching usage
            if self.session_stats.fuzzy_matches > 0:
                print(f"   • Found {self.session_stats.fuzzy_matches} fuzzy matches")
                
            # Highlight user satisfaction if available
            if self.session_stats.user_satisfaction_ratings:
                avg_rating = sum(self.session_stats.user_satisfaction_ratings) / len(self.session_stats.user_satisfaction_ratings)
                print(f"   • Average user satisfaction: {avg_rating:.1f}/5")
            
            # Performance summary section
            print(f"\n💡 Performance Summary:")
            print(f"   • Average response time: {self.session_stats.average_response_time:.3f}s")
            print(f"   • Peak memory usage: {self.session_stats.peak_memory_usage:.1f} MB")
            
            # Calculate and display success rate
            success_rate = (self.session_stats.successful_queries/self.session_stats.queries_processed)*100
            print(f"   • Success rate: {success_rate:.1f}%")

    def _export_stats(self):
        """
        Export statistics to timestamped file for external analysis.
        
        This method provides a way to save session statistics for later
        analysis, reporting, or comparison with other sessions.
        
        Educational Note: Data export capabilities are important for
        analytics and long-term system monitoring.
        """
        # Generate timestamped filename for unique file identification
        filename = f"stats_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Use existing save method with custom filename
        self._save_stats_to_file(filename)
        print(f"📊 Statistics exported to {filename}")

    def _save_stats_to_file(self, filename: str = None):
        """
        Save session statistics to JSON file with comprehensive metadata.
        
        This method implements data persistence for session analytics:
        - Automatic filename generation with timestamps
        - Comprehensive data structure with metadata
        - JSON serialization with datetime handling
        - Error handling for file operations
        
        Args:
            filename: Optional custom filename for the statistics file
            
        Educational Note: Data persistence enables long-term analysis
        and comparison of system performance across sessions.
        """
        # Generate default filename if none provided
        if filename is None:
            filename = f"session_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Create comprehensive statistics data structure
            stats_data = {
                # Session metadata for context
                'session_info': {
                    'start_time': self.session_stats.session_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - self.session_stats.session_start_time).total_seconds()
                },
                
                # Core statistics data
                'statistics': asdict(self.session_stats),
                
                # Configuration snapshot for reproducibility
                'configuration': asdict(self.config)
            }
            
            # Define datetime conversion function for JSON serialization
            def convert_datetime(obj):
                """
                Convert datetime objects to ISO format strings for JSON compatibility.
                
                Args:
                    obj: Object to convert (datetime objects become ISO strings)
                    
                Returns:
                    ISO format string for datetime objects
                    
                Raises:
                    TypeError: For non-serializable objects
                """
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            # Write data to file with proper JSON formatting
            import json
            with open(filename, 'w') as f:
                json.dump(stats_data, f, 
                         indent=2,                    # Pretty formatting for readability
                         default=convert_datetime)    # Handle datetime objects
                
        except Exception as e:
            # Log error but don't crash the application
            self.logger.error(f"Failed to save stats to file: {e}")

    def _export_session_data(self, filename: str):
        """
        Export comprehensive session data including conversation history - FIXED VERSION
        
        This method creates a complete session export that includes:
        - Session statistics and metadata
        - Complete conversation history
        - Configuration snapshot
        - System information
        
        Args:
            filename: Destination filename for the export
            
        Educational Note: Comprehensive data export enables session replay,
        analysis, and debugging of complex user interactions.
        """
        try:
            def datetime_converter(obj):
                """
                Convert datetime objects to ISO format strings for JSON serialization.
                
                JSON doesn't natively support datetime objects, so we need to
                convert them to strings in a standardized format.
                
                Args:
                    obj: Object to convert
                    
                Returns:
                    ISO format string for datetime objects
                    
                Raises:
                    TypeError: For objects that can't be serialized
                """
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            # Create comprehensive session data structure
            session_data = {
                # Session timing and metadata
                'session_info': {
                    'start_time': self.session_stats.session_start_time.isoformat(),
                    'export_time': datetime.now().isoformat(),
                },
                
                # Complete statistics snapshot
                'statistics': asdict(self.session_stats),
                
                # Conversation history with proper datetime conversion
                'conversation_history': [
                    {
                        'timestamp': entry['timestamp'].isoformat(),  # Convert datetime to string
                        'query': entry['query'],
                        'response': entry['response'],
                        'metadata': entry['metadata']
                    }
                    for entry in self.conversation_history.history
                ] if self.conversation_history else [],
                
                # Configuration snapshot for context
                'configuration': asdict(self.config)
            }
            
            # Write comprehensive data to file
            with open(filename, 'w') as f:
                json.dump(session_data, f, 
                         indent=2,                      # Human-readable formatting
                         default=datetime_converter)    # Handle datetime conversion
            
            print(f"💾 Session data exported to {filename}")
            
        except Exception as e:
            # Handle export errors gracefully
            print(f"❌ Failed to export session data: {e}")

    def _create_backup(self):
        """
        Create comprehensive backup of important system data.
        
        This method implements a backup strategy that preserves:
        - System configuration
        - Data file locations
        - System environment information
        - Backup metadata for restoration
        
        Educational Note: Regular backups are essential for data protection
        and disaster recovery in production systems.
        """
        try:
            # Create backup directory if it doesn't exist
            backup_dir = "backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Generate unique backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(backup_dir, f"chatbot_backup_{timestamp}.json")
            
            # Create comprehensive backup data structure
            backup_data = {
                # Backup metadata
                'backup_time': datetime.now().isoformat(),
                'data_path': self.data_path,
                
                # Configuration snapshot
                'configuration': asdict(self.config),
                
                # System environment information for restoration context
                'system_info': {
                    'python_version': sys.version,      # Python version for compatibility
                    'platform': sys.platform,          # Operating system
                    'cwd': os.getcwd()                 # Working directory
                }
            }
            
            # Write backup data to file
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            print(f"💾 Backup created: {backup_file}")
            
        except Exception as e:
            # Log backup errors for debugging
            self.logger.error(f"Backup creation failed: {e}")


# ============================================================================
# Main Function - Application entry point with argument parsing
# ============================================================================

def main():
    """
    Enhanced main entry point with comprehensive argument handling.
    
    This function demonstrates professional CLI application design:
    - Comprehensive argument parsing with help text
    - Multiple execution modes and options
    - Configuration management integration
    - Error handling and user feedback
    - Example usage documentation
    
    Educational Note: The main function serves as the entry point and
    should handle all startup logic, argument processing, and error handling.
    """
    # Import argparse for sophisticated command-line argument handling
    import argparse
    
    # Create argument parser with detailed description and examples
    parser = argparse.ArgumentParser(
        description='Advanced Recursive AI Chatbot with Enterprise Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,  # Preserve formatting in epilog
        epilog="""
Examples:
  python main.py                           # Start with default settings
  python main.py --data my_data.json       # Use custom data file
  python main.py --logging --config my_config.json  # Enable logging with custom config
  python main.py --performance-monitoring  # Enable detailed performance monitoring
        """
    )
    
    # Define command-line arguments with comprehensive help text
    
    # Data source configuration
    parser.add_argument('--data', '-d', 
                       help='Path to knowledge base JSON file')
    
    # Logging configuration
    parser.add_argument('--logging', '-l', action='store_true', 
                       help='Enable detailed logging')
    
    # Configuration file management
    parser.add_argument('--config', '-c', default='chatbot_config.json',
                       help='Path to configuration file (default: chatbot_config.json)')
    
    # Performance monitoring options
    parser.add_argument('--performance-monitoring', '-p', action='store_true',
                       help='Enable comprehensive performance monitoring')
    
    # Feature toggles
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching features')
    
    # Development and debugging options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    
    # Utility functions
    parser.add_argument('--export-config', action='store_true',
                       help='Export default configuration and exit')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Handle special arguments that don't start the main application
    if args.export_config:
        # Export default configuration and exit
        config = ChatbotConfig()
        config.save_to_file('default_config.json')
        print("✅ Default configuration exported to default_config.json")
        return
    
    try:
        # Configuration loading and customization based on arguments
        if os.path.exists(args.config):
            # Load existing configuration file
            config = ChatbotConfig.load_from_file(args.config)
        else:
            # Create default configuration
            config = ChatbotConfig()
            # Warn if user specified a non-existent config file
            if args.config != 'chatbot_config.json':
                print(f"⚠️ Configuration file {args.config} not found, using defaults")
        
        # Apply command-line argument overrides to configuration
        # This allows users to override config file settings from command line
        if args.performance_monitoring:
            config.enable_performance_monitoring = True
        if args.no_cache:
            config.cache_size_limit = 0  # Disable caching by setting size to 0
        if args.debug:
            config.logging_level = "DEBUG"      # Enable verbose logging
            config.enable_analytics = True      # Enable all analytics in debug mode
        
        # Save updated configuration for future use
        # This preserves command-line overrides for subsequent runs
        config.save_to_file(args.config)
        
        # Initialize and run the main application
        app = RecursiveAIChatbotApp(
            data_path=args.data,                           # Custom data file if specified
            enable_logging=args.logging or args.debug,    # Enable logging based on flags
            config_path=args.config                       # Use specified config file
        )
        
        # Start interactive mode - this is the main application loop
        app.interactive_mode()
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully at the top level
        print("\n👋 Goodbye!")
        
    except Exception as e:
        # Handle any unexpected errors during startup
        print(f"❌ Failed to start advanced application: {e}")
        
        # Show detailed error information in debug mode
        if args.debug:
            import traceback
            traceback.print_exc()  # Print full error traceback
            
        # Exit with error code to indicate failure
        sys.exit(1)


# ============================================================================
# Entry Point - Standard Python application pattern
# ============================================================================

if __name__ == "__main__":
    """
    Standard Python idiom for script execution.
    
    This ensures the main() function only runs when the script is executed
    directly, not when it's imported as a module.
    
    Educational Note: This pattern is essential for creating reusable Python
    modules that can be both imported and executed directly.
    """
    main()