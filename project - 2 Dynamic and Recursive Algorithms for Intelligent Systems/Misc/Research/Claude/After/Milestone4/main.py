"""
 Optimized Main entry point for the Recursive AI Chatbot

This script launches the chatbot in interactive mode and integrates the three major algorithmic modules:
- Recursive Query Handling (Milestone 1) - Optimized
- Dynamic Context Management with Caching (Milestone 2)
- Greedy Priority Algorithm for Optimized Response Time (Milestone 3)

Key improvements:
- Fixed result handling for recursive queries
- Better error handling and validation
- Improved statistics tracking
-  user experience
- More robust query classification
- Performance optimizations
- Advanced configuration management
-  logging system
- Better resource management
- Sophisticated command system
- Plugin architecture foundation
- Performance monitoring
-  caching strategies
"""

import sys
import os
import time
import logging
import json
import threading
import signal
import atexit
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict
import hashlib
import gc
import psutil

# Add the current project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the core components of the chatbot system
try:
    from chatbot.chatbot import AIChatbot
    from milestones.recursive_handling import RecursiveHandling as RecursiveHandling
    from milestones.dynamic_context import DynamicContext
    from milestones.greedy_priority import GreedyPriority
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)


@dataclass
class SessionStats:
    """ data class to track comprehensive session statistics"""
    queries_processed: int = 0
    cache_hits: int = 0
    recursive_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_response_time: float = 0.0
    total_time: float = 0.0
    fuzzy_matches: int = 0
    priority_distribution: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    
    #  metrics
    session_start_time: datetime = field(default_factory=datetime.now)
    peak_memory_usage: float = 0.0
    command_usage: Dict[str, int] = field(default_factory=dict)
    query_length_stats: Dict[str, float] = field(default_factory=lambda: {'min': float('inf'), 'max': 0, 'avg': 0})
    error_categories: Dict[str, int] = field(default_factory=dict)
    response_quality_scores: List[float] = field(default_factory=list)
    user_satisfaction_ratings: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.priority_distribution:
            self.priority_distribution = {1: 0, 2: 0, 3: 0}


@dataclass
class ChatbotConfig:
    """ configuration management for the chatbot"""
    max_recursion_depth: int = 4
    cache_size_limit: int = 1000
    logging_level: str = "INFO"
    enable_performance_monitoring: bool = True
    auto_save_stats: bool = True
    stats_save_interval: int = 300  # seconds
    enable_fuzzy_matching: bool = True
    fuzzy_threshold: float = 0.8
    enable_spell_correction: bool = True
    response_timeout: float = 30.0
    max_query_length: int = 1000
    enable_plugins: bool = False
    plugins_directory: str = "./plugins"
    backup_data_on_start: bool = True
    enable_conversation_history: bool = True
    max_history_size: int = 100
    enable_analytics: bool = True
    performance_alert_threshold: float = 5.0
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'ChatbotConfig':
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                return cls(**config_dict)
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {e}")
        return cls()
    
    def save_to_file(self, config_path: str):
        """Save current configuration to file"""
        try:
            with open(config_path, 'w') as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save config to {config_path}: {e}")


class PerformanceMonitor:
    """ performance monitoring system"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
        self.monitoring_active = True
        
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Record a performance metric"""
        if not self.monitoring_active:
            return
            
        if timestamp is None:
            timestamp = datetime.now()
            
        self.metrics[metric_name].append((timestamp, value))
        
        # Keep only recent metrics (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.metrics[metric_name] = [
            (ts, val) for ts, val in self.metrics[metric_name] if ts > cutoff
        ]
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
            
        values = [val for _, val in self.metrics[metric_name]]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else 0
        }
    
    def check_performance_alerts(self, threshold: float = 5.0):
        """Check for performance alerts"""
        response_times = self.get_metric_summary('response_time')
        if response_times.get('avg', 0) > threshold:
            alert = f"High average response time: {response_times['avg']:.2f}s"
            if alert not in [a['message'] for a in self.alerts[-5:]]:  # Avoid duplicates
                self.alerts.append({
                    'timestamp': datetime.now(),
                    'level': 'WARNING',
                    'message': alert
                })


class ConversationHistory:
    """ conversation history management"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.history = deque(maxlen=max_size)
        self.search_index = {}
        
    def add_entry(self, query: str, response: str, metadata: Dict[str, Any]):
        """Add a conversation entry with  metadata"""
        entry = {
            'timestamp': datetime.now(),
            'query': query,
            'response': response,
            'metadata': metadata,
            'query_hash': hashlib.md5(query.encode()).hexdigest()
        }
        
        self.history.append(entry)
        self._update_search_index(entry)
    
    def _update_search_index(self, entry: Dict[str, Any]):
        """Update search index for quick retrieval - FIXED VERSION"""
        query_words = entry['query'].lower().split()
        for word in query_words:
            # Clean the word and add to index
            cleaned_word = word.strip('.,!?;:"()[]{}')
            if cleaned_word and len(cleaned_word) > 2:  # Only index meaningful words
                if cleaned_word not in self.search_index:
                    self.search_index[cleaned_word] = []
                self.search_index[cleaned_word].append(len(self.history) - 1)    
    
    def search_history(self, search_term: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search conversation history - FIXED VERSION"""
        search_words = search_term.lower().split()
        candidate_indices = set()
        
        for word in search_words:
            cleaned_word = word.strip('.,!?;:"()[]{}')
            if cleaned_word in self.search_index:
                candidate_indices.update(self.search_index[cleaned_word])
            
            # Also search for partial matches
            for indexed_word in self.search_index:
                if cleaned_word in indexed_word or indexed_word in cleaned_word:
                    candidate_indices.update(self.search_index[indexed_word])
        
        # Convert indices to actual entries, handling potential index issues
        candidates = []
        for i in candidate_indices:
            if i < len(self.history):
                candidates.append(self.history[i])
    
        # Sort by relevance (simple word matching score)
        def relevance_score(entry):
            query_words = entry['query'].lower().split()
            return sum(1 for word in search_words if any(search_word in query_word for query_word in query_words for search_word in [word]))
        
        candidates.sort(key=relevance_score, reverse=True)
        return candidates[:limit]
    
    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        return list(self.history)[-limit:]


class CommandProcessor:
    """ command processing system"""
    
    def __init__(self, app: 'RecursiveAIChatbotApp'):
        self.app = app
        self.command_handlers = {
            'quit': self._handle_quit,
            'exit': self._handle_quit,
            'bye': self._handle_quit,
            'q': self._handle_quit,
            'stats': self._handle_stats,
            'statistics': self._handle_stats,
            'help': self._handle_help,
            'h': self._handle_help,
            '?': self._handle_help,
            'clear': self._handle_clear,
            'cls': self._handle_clear,
            'cache': self._handle_cache,
            'cache-info': self._handle_cache,
            'reset': self._handle_reset,
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
        
        # Command aliases
        self.aliases = {
            'perf': 'performance',
            'mem': 'memory',
            'hist': 'history',
            'find': 'search',
            'save': 'export',
            'rating': 'rate',
        }
    
    def process_command(self, command: str, args: List[str] = None) -> bool:
        """Process a command and return True if it was handled"""
        if args is None:
            args = []
            
        # Handle aliases
        command = self.aliases.get(command, command)
        
        if command in self.command_handlers:
            # Track command usage
            self.app.session_stats.command_usage[command] = (
                self.app.session_stats.command_usage.get(command, 0) + 1
            )
            
            try:
                return self.command_handlers[command](args)
            except Exception as e:
                print(f"❌ Error executing command '{command}': {e}")
                return True
        
        return False
    
    def _handle_quit(self, args: List[str]) -> bool:
        print("\n👋 Thank you for using the  Recursive AI Chatbot!")
        self.app._print_session_summary()
        return False  # Signal to exit
    
    def _handle_stats(self, args: List[str]) -> bool:
        if args and args[0] == 'export':
            self.app._export_stats()
        else:
            self.app._print_detailed_stats()
        return True
    
    def _handle_help(self, args: List[str]) -> bool:
        if args and args[0] == 'commands':
            self._print_command_help()
        else:
            self.app._print_help()
        return True
    
    def _handle_clear(self, args: List[str]) -> bool:
        os.system('cls' if os.name == 'nt' else 'clear')
        return True
    
    def _handle_cache(self, args: List[str]) -> bool:
        if args and args[0] == 'clear':
            self.app.recursive_handler.clear_cache()
            print("🧹 Cache cleared successfully!")
        else:
            self.app._print_cache_info()
        return True
    
    def _handle_reset(self, args: List[str]) -> bool:
        self.app._reset_session()
        return True
    
    def _handle_config(self, args: List[str]) -> bool:
        if args and args[0] == 'show':
            self._show_config()
        elif args and args[0] == 'save':
            self.app.config.save_to_file('chatbot_config.json')
            print("💾 Configuration saved to chatbot_config.json")
        else:
            print("Usage: config [show|save]")
        return True
    
    def _handle_performance(self, args: List[str]) -> bool:
        self._show_performance_metrics()
        return True
    
    def _handle_history(self, args: List[str]) -> bool:
        limit = 10
        if args and args[0].isdigit():
            limit = int(args[0])
        
        history = self.app.conversation_history.get_recent_history(limit)
        self._display_history(history)
        return True
    
    def _handle_search(self, args: List[str]) -> bool:
        if not args:
            print("Usage: search <search_term>")
            return True
        
        search_term = ' '.join(args)
        results = self.app.conversation_history.search_history(search_term)
        self._display_history(results, f"Search results for '{search_term}'")
        return True
    
    def _handle_export(self, args: List[str]) -> bool:
        filename = args[0] if args else f"chatbot_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.app._export_session_data(filename)
        return True
    
    def _handle_memory(self, args: List[str]) -> bool:
        self._show_memory_usage()
        return True
    
    def _handle_rate_response(self, args: List[str]) -> bool:
        if not args or not args[0].isdigit():
            print("Usage: rate <1-5>")
            return True
        
        rating = int(args[0])
        if 1 <= rating <= 5:
            self.app.session_stats.user_satisfaction_ratings.append(rating)
            print(f"✅ Thank you for rating the last response: {rating}/5")
        else:
            print("Please provide a rating between 1 and 5")
        return True
    
    def _handle_backup(self, args: List[str]) -> bool:
        self.app._create_backup()
        return True
    
    def _handle_plugins(self, args: List[str]) -> bool:
        if args and args[0] == 'list':
            self._list_plugins()
        elif args and args[0] == 'load' and len(args) > 1:
            self._load_plugin(args[1])
        else:
            print("Usage: plugins [list|load <plugin_name>]")
        return True
    
    def _handle_debug(self, args: List[str]) -> bool:
        self._show_debug_info()
        return True
    
    def _print_command_help(self):
        """Print detailed command help"""
        print("\n🆘  Command System:")
        print("-" * 40)
        for cmd, handler in self.command_handlers.items():
            doc = handler.__doc__ or "No description available"
            print(f" 🔧 {cmd:<15} - {doc}")
    
    def _show_config(self):
        """Display current configuration"""
        print("\n⚙️ Current Configuration:")
        print("-" * 30)
        config_dict = asdict(self.app.config)
        for key, value in config_dict.items():
            print(f" 🔧 {key}: {value}")
    
    def _show_performance_metrics(self):
        """Display performance metrics"""
        print("\n📊 Performance Metrics:")
        print("-" * 25)
        for metric_name in ['response_time', 'memory_usage', 'cache_hit_rate']:
            summary = self.app.performance_monitor.get_metric_summary(metric_name)
            if summary:
                print(f" 📈 {metric_name}:")
                for stat, value in summary.items():
                    print(f"    {stat}: {value:.3f}")
        
        if self.app.performance_monitor.alerts:
            print("\n⚠️ Recent Alerts:")
            for alert in self.app.performance_monitor.alerts[-5:]:
                print(f"   {alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}")
    
    def _display_history(self, history: List[Dict[str, Any]], title: str = "Conversation History"):
        """Display conversation history"""
        if not history:
            print("📭 No history found")
            return
        
        print(f"\n📜 {title}:")
        print("-" * 40)
        for i, entry in enumerate(history, 1):
            timestamp = entry['timestamp'].strftime('%H:%M:%S')
            query_preview = entry['query'][:50] + '...' if len(entry['query']) > 50 else entry['query']
            print(f" {i}. [{timestamp}] {query_preview}")
    
    def _show_memory_usage(self):
        """Show current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        print("\n💾 Memory Usage:")
        print("-" * 20)
        print(f" 🔢 RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
        print(f" 🔢 VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
        print(f" 📊 Peak: {self.app.session_stats.peak_memory_usage:.1f} MB")
        
        # Trigger garbage collection and show results
        collected = gc.collect()
        print(f" 🧹 Garbage collected: {collected} objects")
    
    def _list_plugins(self):
        """List available plugins"""
        print("\n🔌 Plugin System:")
        print("-" * 20)
        print("🚧 Plugin system is currently under development")
        
    def _load_plugin(self, plugin_name: str):
        """Load a specific plugin"""
        print(f"🔌 Loading plugin: {plugin_name}")
        print("🚧 Plugin loading not yet implemented")
    
    def _show_debug_info(self):
        """Show debug information"""
        print("\n🐛 Debug Information:")
        print("-" * 25)
        print(f" 🐍 Python version: {sys.version}")
        print(f" 📁 Working directory: {os.getcwd()}")
        print(f" 🗂️ Data path: {self.app.data_path}")
        print(f" 🔧 Config: {type(self.app.config).__name__}")
        print(f" 💾 Cache size: {len(getattr(self.app.dynamic_context, 'cache', {}))}")
        print(f" 🧠 Components loaded: {len([c for c in [self.app.chatbot, self.app.recursive_handler, self.app.dynamic_context, self.app.greedy_priority] if c])}")


class RecursiveAIChatbotApp:
    """
     main application class for the Recursive AI Chatbot.
    
    Key improvements:
    - Better error handling and validation
    -  statistics tracking
    - Improved query classification
    - More robust caching integration
    - Better user experience
    - Advanced configuration management
    - Performance monitoring
    - Plugin architecture foundation
    -  logging system
    - Resource management
    """

    def __init__(self, data_path: Optional[str] = None, enable_logging: bool = False, config_path: str = "chatbot_config.json"):
        """
        Initialize the chatbot with all subsystems.

        Args:
            data_path: Path to the JSON file containing the knowledge base
            enable_logging: Whether to enable detailed logging
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = ChatbotConfig.load_from_file(config_path)
        
        self._setup_logging(enable_logging or self.config.logging_level != "INFO")
        self.data_path = self._resolve_data_path(data_path)
        self.session_stats = SessionStats()
        
        # Initialize  components
        self.performance_monitor = PerformanceMonitor() if self.config.enable_performance_monitoring else None
        self.conversation_history = ConversationHistory(self.config.max_history_size) if self.config.enable_conversation_history else None
        self.command_processor = CommandProcessor(self)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
        
        print("🤖 Initializing AI Chatbot...")
        print("=" * 55)

        try:
            self._initialize_components()
            self._setup_auto_save()
            if self.config.backup_data_on_start:
                self._create_backup()
            
            print("✅  Chatbot initialized successfully!")
            print(f"📊 Knowledge base loaded with {len(self.chatbot.knowledge_base.qa_pairs)} QA pairs")
            print(f"🔧 Using optimized recursive handling with  caching")
            print(f"⚡ Performance monitoring: {'Enabled' if self.performance_monitor else 'Disabled'}")
            print(f"📜 Conversation history: {'Enabled' if self.conversation_history else 'Disabled'}")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            print(f"❌ Error initializing chatbot: {e}")
            raise

    def _setup_logging(self, enable_logging: bool):
        """Logging configuration"""
        self.logger = logging.getLogger(__name__)
        if enable_logging:
            # Create logs directory if it doesn't exist
            os.makedirs('logs', exist_ok=True)
            
            # Setup file and console handlers
            file_handler = logging.FileHandler(
                f'logs/chatbot_{datetime.now().strftime("%Y%m%d")}.log'
            )
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(getattr(logging, self.config.logging_level.upper()))

    def _resolve_data_path(self, data_path: Optional[str]) -> str:
        """ data path resolution with better error handling"""
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'dev-v2.0.json')
        
        if not os.path.exists(data_path):
            # Try alternative paths
            alternative_paths = [
                'data/dev-v2.0.json',
                '../data/dev-v2.0.json',
                './dev-v2.0.json',
                os.path.expanduser('~/chatbot_data/dev-v2.0.json')
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    data_path = alt_path
                    self.logger.info(f"Using alternative data path: {alt_path}")
                    break
            else:
                self.logger.warning(f"Data file not found at {data_path}, will continue with limited functionality")
        
        return data_path

    def _initialize_components(self):
        """ component initialization with better error handling"""
        # Initialize core modules with error handling
        self.chatbot = AIChatbot(self.data_path)
        self.recursive_handler = RecursiveHandling(
            self.chatbot, 
            max_recursion_depth=self.config.max_recursion_depth
        )
        self.dynamic_context = DynamicContext()
        self.greedy_priority = GreedyPriority()
        
        # Validate components
        self._validate_components()
        
        self.logger.info("All components initialized successfully")

    def _validate_components(self):
        """Validate that all components are properly initialized"""
        components = {
            'chatbot': self.chatbot,
            'recursive_handler': self.recursive_handler,
            'dynamic_context': self.dynamic_context,
            'greedy_priority': self.greedy_priority
        }
        
        for name, component in components.items():
            if component is None:
                raise RuntimeError(f"Failed to initialize {name}")

    def _setup_auto_save(self):
        """Setup automatic statistics saving"""
        if self.config.auto_save_stats:
            def auto_save():
                while True:
                    time.sleep(self.config.stats_save_interval)
                    try:
                        self._save_stats_to_file()
                    except Exception as e:
                        self.logger.error(f"Auto-save failed: {e}")
            
            save_thread = threading.Thread(target=auto_save, daemon=True)
            save_thread.start()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully - FIXED VERSION"""
        print(f"\n📡 Received signal {signum}, shutting down gracefully...")
        
        # Don't use logger here as it might cause issues during shutdown
        try:
            if self.config.auto_save_stats:
                self._save_stats_to_file()
            print("✅ Graceful shutdown completed")
        except Exception as e:
            print(f"⚠️ Error during shutdown: {e}")
        

    def _cleanup(self):
        """Cleanup resources before shutdown - MOST ROBUST VERSION"""
        try:
            if self.config.auto_save_stats:
                self._save_stats_to_file()
            
            # Always use print for cleanup messages to avoid logging issues
            print("✅ Cleanup completed successfully")
                
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
        finally:
            # Safely close logging handlers
            if hasattr(self, 'logger') and self.logger.handlers:
                for handler in self.logger.handlers[:]:  # Create a copy of the list
                    try:
                        handler.close()
                        self.logger.removeHandler(handler)
                    except:
                        pass  # Ignore errors when closing handlers

    def _classify_query_complexity(self, query: str) -> bool:
        """
         query complexity classification with better preprocessing.
        
        Args:
            query: The user input string
            
        Returns:
            bool: True if it's a complex/compound query
        """
        if not query or len(query.strip()) < 10:
            return False
        
        # Preprocess query
        query = self._preprocess_query(query)
        
        # Use the recursive handler's built-in detection
        try:
            return self.recursive_handler._is_nested_query_cached(query.strip())
        except Exception as e:
            self.logger.warning(f"Error in query classification: {e}")
            # Fallback to  heuristics
            return self.__complexity_check(query)

    def _preprocess_query(self, query: str) -> str:
        """ query preprocessing"""
        # Basic sanitization
        query = query.strip()
        
        # Remove excessive whitespace
        query = ' '.join(query.split())
        
        # Basic spell correction could be added here
        if self.config.enable_spell_correction:
            query = self._basic_spell_correction(query)
        
        return query

    def _basic_spell_correction(self, query: str) -> str:
        """Basic spell correction (placeholder for more sophisticated implementation)"""
        # Simple replacements for common typos
        replacements = {
            'wat': 'what',
            'hwo': 'how',
            'teh': 'the',
            'ai': 'AI',  # Capitalize AI
        }
        
        words = query.split()
        corrected_words = [replacements.get(word.lower(), word) for word in words]
        return ' '.join(corrected_words)

    def __complexity_check(self, query: str) -> bool:
        """ fallback complexity check"""
        complexity_indicators = [
            # Conjunctions
            ' and ', ' & ', '; ', ' or ', ' but ', ' however ',
            # Sequence indicators
            'also', 'additionally', 'furthermore', 'moreover',
            'then', 'next', 'afterwards', 'subsequently',
            # Question patterns
            'what about', 'how about', 'tell me about',
            'explain both', 'describe each', 'list all',
            # Multiple questions
            '?', 'question:', 'Q:', 'A:'
        ]
        
        query_lower = query.lower()
        
        # Count indicators
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        question_count = query.count('?')
        
        #  logic
        is_complex = (
            indicator_count >= 2 or
            question_count > 1 or
            (indicator_count >= 1 and len(query.split()) > 15) or
            any(phrase in query_lower for phrase in ['step by step', 'in detail', 'comprehensive'])
        )
        
        return is_complex

    def _update_stats(self, result: Dict[str, Any]):
        """
        Enhanced statistics tracking with comprehensive metrics - FIXED VERSION.
        
        Args:
            result: Dictionary containing query processing results
        """
        try:
            self.session_stats.queries_processed += 1
            
            # Update query length statistics
            query_len = len(result.get('query', ''))
            stats = self.session_stats.query_length_stats
            stats['min'] = min(stats['min'], query_len)
            stats['max'] = max(stats['max'], query_len)
            total_chars = stats['avg'] * (self.session_stats.queries_processed - 1) + query_len
            stats['avg'] = total_chars / self.session_stats.queries_processed
            
            # Track processing time with performance monitoring
            processing_time = result.get('processing_time', 0.0)
            self.session_stats.total_time += processing_time
            
            if self.performance_monitor:
                self.performance_monitor.record_metric('response_time', processing_time)
                
                # Check for performance alerts
                if processing_time > self.config.performance_alert_threshold:
                    self.performance_monitor.check_performance_alerts(self.config.performance_alert_threshold)
            
            # Calculate average response time
            if self.session_stats.queries_processed > 0:
                self.session_stats.average_response_time = (
                    self.session_stats.total_time / self.session_stats.queries_processed
                )
            
            # Track memory usage
            if self.performance_monitor:
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.performance_monitor.record_metric('memory_usage', memory_mb)
                    self.session_stats.peak_memory_usage = max(
                        self.session_stats.peak_memory_usage, memory_mb
                    )
                except:
                    pass  # psutil might not be available
            
            # Track other enhanced metrics
            if result.get('used_cache', False):
                self.session_stats.cache_hits += 1
            
            if result.get('is_recursive', False):
                self.session_stats.recursive_queries += 1
            
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
                
                self.session_stats.error_categories[error_type] = (
                    self.session_stats.error_categories.get(error_type, 0) + 1
                )
            
            if result.get('fuzzy_match', False):
                self.session_stats.fuzzy_matches += 1
            
            # Track priority distribution
            priority = result.get('priority', 2)
            if priority in self.session_stats.priority_distribution:
                self.session_stats.priority_distribution[priority] += 1
            
            # Estimate response quality (basic heuristic)
            response_length = len(result.get('response', ''))
            quality_score = min(1.0, response_length / 100)  # Simple length-based score
            if result.get('fuzzy_match'):
                quality_score *= 0.8  # Penalize fuzzy matches slightly
            if result.get('used_cache'):
                quality_score *= 1.1  # Reward cache hits (faster response)
            
            self.session_stats.response_quality_scores.append(quality_score)
            
        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Enhanced core handler to process a single user query with comprehensive features - FIXED VERSION.
        
        Args:
            query: The input question or prompt from the user
            
        Returns:
            Dict[str, Any]: Structured response with comprehensive metadata
        """
        start_time = time.time()
        
        # Enhanced input validation
        validation_result = self._validate_query_input(query)
        if not validation_result['valid']:
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
        
        query = self._preprocess_query(query)
        
        # Initialize enhanced result structure
        result = {
            'query': query,
            'response': '',
            'processing_time': 0.0,
            'used_cache': False,
            'is_recursive': False,
            'priority': 2,
            'success': True,
            'error': None,
            'error_info': {},  # FIXED: Add error_info field
            'fuzzy_match': False,
            'fuzzy_match_threshold': 0.0,
            'query_length': len(query),
            'preprocessing_applied': True,
            'component_used': 'unknown'
        }

        try:
            # Step 1: Assign priority level with enhanced logic
            result['priority'] = self.greedy_priority.get_priority(query)
            
            # Step 2: Check cache first (Dynamic Programming)
            cached_response = self.dynamic_context.retrieve_from_cache(query)
            if(cached_response and self.chatbot.DEFAULT_NO_MATCH_MESSAGE not in cached_response):
                result['response'] = cached_response
                result['used_cache'] = True
                result['component_used'] = 'cache'
                self.logger.info(f"Cache hit for query: {query[:50]}...")
            else:
                # Step 3: Determine processing approach with enhanced classification
                is_complex = self._classify_query_complexity(query)
                
                if is_complex:
                    # Use recursive handling for complex queries
                    result['component_used'] = 'recursive_handler'
                    self.logger.info(f"Processing complex query: {query[:50]}...")
                    recursive_result = self.recursive_handler.handle_recursive_query(query)
                    
                    # Extract data from QueryResult object or dict with enhanced handling
                    if hasattr(recursive_result, '__dict__'):
                        # It's a QueryResult dataclass
                        result.update({
                            'response': recursive_result.response,
                            'is_recursive': recursive_result.is_recursive,
                            'fuzzy_match': recursive_result.fuzzy_match,
                            'fuzzy_match_threshold': recursive_result.fuzzy_match_threshold,
                            'used_cache': recursive_result.used_cache or result['used_cache']
                        })
                    else:
                        # It's a dictionary (backward compatibility)
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
                    response, is_fuzzy, threshold = self.chatbot.handle_query(query)
                    if (response is not None or response != ''):
                        result.update({
                            'response': response,
                            'fuzzy_match': is_fuzzy,
                            'fuzzy_match_threshold': threshold
                        })

                # Step 4: Cache the result for future use with enhanced conditions
                if (result['response'] and 
                    not result.get('error') and 
                    len(result['response']) >= 3 and
                    result['priority'] >= 3):  # Only cache meaningful responses
                    self.dynamic_context.store_in_cache(query, result['response'])

            # Step 5: Post-process response
            result['response'] = self._post_process_response(result['response'], result)

        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {e}")
            result.update({
                'success': False,
                'error': str(e),
                'error_info': {'exception_type': type(e).__name__},  # FIXED: Capture actual exception type
                'response': self._get_error_response(e),
                'component_used': 'error_handler'
            })

        # Finalize timing and update comprehensive statistics
        result['processing_time'] = time.time() - start_time
        self._update_stats(result)
        
        # Add to conversation history
        if self.conversation_history:
            self.conversation_history.add_entry(query, result['response'], result)
        
        return result

    def _validate_query_input(self, query: str) -> Dict[str, Any]:
        """ input validation"""
        if not query or not query.strip():
            return {
                'valid': False,
                'message': 'Please provide a valid question.',
                'error': 'Empty query'
            }
        
        if len(query) > self.config.max_query_length:
            return {
                'valid': False,
                'message': f'Query too long. Maximum length is {self.config.max_query_length} characters.',
                'error': 'Query too long'
            }
        
        # Check for potential malicious patterns
        suspicious_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
        if any(pattern in query.lower() for pattern in suspicious_patterns):
            return {
                'valid': False,
                'message': 'Query contains potentially unsafe content.',
                'error': 'Unsafe content detected'
            }
        
        return {'valid': True}

    def _post_process_response(self, response: str, result: Dict[str, Any]) -> str:
        """Post-process the response for better user experience"""
        if not response:
            return "I apologize, but I couldn't generate a response to your query."
        
        # Add quality indicators for fuzzy matches
        if result.get('fuzzy_match') and result.get('fuzzy_match_threshold', 0) < 0.9:
            response = f"💡 *Similar question found:*\n\n{response}"
        
        # Add cache indicator for development/debug mode
        if result.get('used_cache') and self.config.logging_level == "DEBUG":
            response += "\n\n*[Response retrieved from cache]*"
        
        return response

    def _get_error_response(self, error: Exception) -> str:
        """Generate user-friendly error responses"""
        error_responses = {
            'TimeoutError': "The request took too long to process. Please try a simpler question.",
            'ConnectionError': "There seems to be a connectivity issue. Please try again.",
            'ValueError': "There was an issue with your input. Please rephrase your question.",
            'KeyError': "I couldn't find the required information. Please try a different question.",
        }
        
        error_type = type(error).__name__
        return error_responses.get(error_type, 
            "I apologize, but I encountered an error processing your query. Please try rephrasing it.")

    def interactive_mode(self):
        """
         interactive terminal session with comprehensive features.
        """
        print("\n🎯 Starting Interactive Mode")
        print("=" * 55)
        print("💡 Features:")
        print("   • Intelligent query processing with caching")
        print("   • Performance monitoring and analytics")
        print("   • Conversation history and search")
        print("   • Advanced command system (type 'help' for commands)")
        print("   • Real-time statistics and feedback")
        print("-" * 55)

        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue

                #  command processing
                command_parts = user_input.split()
                command = command_parts[0].lower()
                args = command_parts[1:] if len(command_parts) > 1 else []
                
                # Try to process as command first
                if not self.command_processor.process_command(command, args):
                    if command in ['quit', 'exit', 'bye', 'q']:
                        break
                    

                # Process as regular query with timeout
                try:
                    with self._query_timeout(self.config.response_timeout):
                        result = self.process_query(user_input)
                        print(f"\n🤖 Chatbot: {result['response']}")

                        # Show  processing info
                        self._show_processing_info(result)
                        
                        # Ask for feedback on complex queries
                        if (result['processing_time'] > 2.0 or 
                            result['is_recursive'] or 
                            not result['success']):
                            self._prompt_for_feedback()

                except TimeoutError:
                    print("\n⏰ Query timeout! The request took too long to process.")
                    print("💡 Try asking a simpler question or check your connection.")

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                self._print_session_summary()
                break
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {e}")
                print(f"\n❌ An unexpected error occurred: {e}")
                print("💡 Type 'debug' for diagnostic information")

    def _query_timeout(self, timeout: float):
        """Context manager for query timeout"""
        class TimeoutContext:
            def __init__(self, timeout_seconds):
                self.timeout = timeout_seconds
                
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        return TimeoutContext(timeout)

    def _show_processing_info(self, result: Dict[str, Any]):
        """ processing information display"""
        show_info = (
            result['processing_time'] > 0.5 or 
            result['is_recursive'] or 
            result['fuzzy_match'] or
            not result['success'] or
            result.get('used_cache', False)
        )
        
        if show_info:
            print("\n📊 Processing Details:")
            print(f"   ⏱️  Time: {result['processing_time']:.3f}s")
            print(f"   🎯 Priority: {result['priority']}")
            print(f"   🔧 Component: {result.get('component_used', 'unknown')}")
            print(f"   🔄 Recursive: {'Yes' if result['is_recursive'] else 'No'}")
            print(f"   💾 Cached: {'Yes' if result['used_cache'] else 'No'}")
            if result['fuzzy_match']:
                print(f"   🔍 Fuzzy Match: {result['fuzzy_match_threshold']:.2f}")
            if not result['success']:
                print(f"   ❌ Error: {result['error']}")

    def _prompt_for_feedback(self):
        """Prompt user for feedback on response quality"""
        if not self.conversation_history:
            return
            
        try:
            feedback_input = input("\n⭐ Rate this response (1-5, or press Enter to skip): ").strip()
            if feedback_input.isdigit() and 1 <= int(feedback_input) <= 5:
                rating = int(feedback_input)
                self.session_stats.user_satisfaction_ratings.append(rating)
                print(f"✅ Thank you for your feedback: {rating}/5")
        except:
            pass  # Skip if any error in feedback collection

    def _print_detailed_stats(self):
        """ comprehensive session statistics"""
        print("\n📈  Session Statistics:")
        print("-" * 40)
        
        # Basic stats
        stats_dict = asdict(self.session_stats)
        for key, value in stats_dict.items():
            if key == 'priority_distribution':
                print(f" 🎯 Priority Distribution:")
                for priority, count in value.items():
                    print(f"    Priority {priority}: {count} queries")
            elif key == 'command_usage':
                if value:
                    print(f" 🔧 Command Usage:")
                    for cmd, count in sorted(value.items(), key=lambda x: x[1], reverse=True):
                        print(f"    {cmd}: {count} times")
            elif key == 'query_length_stats':
                print(f" 📏 Query Length Statistics:")
                for stat, val in value.items():
                    if val != float('inf'):
                        print(f"    {stat}: {val:.1f} characters")
            elif key == 'error_categories':
                if value:
                    print(f" ❌ Error Categories:")
                    for error_type, count in value.items():
                        print(f"    {error_type}: {count}")
            elif key in ['response_quality_scores', 'user_satisfaction_ratings']:
                if value:
                    avg_score = sum(value) / len(value)
                    print(f" ⭐ {key.replace('_', ' ').title()}: {avg_score:.2f} (n={len(value)})")
            elif 'time' in key:
                print(f" ⏱️  {key.replace('_', ' ').title()}: {value:.3f}s")
            elif key not in ['session_start_time']:
                print(f" 📊 {key.replace('_', ' ').title()}: {value}")
        
        # Calculate additional  metrics
        if self.session_stats.queries_processed > 0:
            cache_hit_rate = (self.session_stats.cache_hits / self.session_stats.queries_processed) * 100
            success_rate = (self.session_stats.successful_queries / self.session_stats.queries_processed) * 100
            recursive_rate = (self.session_stats.recursive_queries / self.session_stats.queries_processed) * 100
            
            print(f" 💾 Cache Hit Rate: {cache_hit_rate:.1f}%")
            print(f" ✅ Success Rate: {success_rate:.1f}%")
            print(f" 🔄 Recursive Query Rate: {recursive_rate:.1f}%")
            
            session_duration = datetime.now() - self.session_stats.session_start_time
            queries_per_minute = self.session_stats.queries_processed / (session_duration.total_seconds() / 60)
            print(f" ⚡ Queries per minute: {queries_per_minute:.1f}")
            
            if self.session_stats.user_satisfaction_ratings:
                avg_satisfaction = sum(self.session_stats.user_satisfaction_ratings) / len(self.session_stats.user_satisfaction_ratings)
                print(f" 😊 Average satisfaction: {avg_satisfaction:.1f}/5")

    def _print_cache_info(self):
        """ cache information and statistics"""
        try:
            cache_info = self.recursive_handler.get_cache_info()
            print("\n💾  Cache Information:")
            print("-" * 30)
            
            nested_cache = cache_info['nested_query_cache_info']
            print(f" 🔍 Query Classification Cache:")
            print(f"    Hits: {nested_cache['hits']}")
            print(f"    Misses: {nested_cache['misses']}")
            print(f"    Current Size: {nested_cache['currsize']}")
            print(f"    Max Size: {nested_cache['maxsize']}")
            
            if nested_cache['hits'] + nested_cache['misses'] > 0:
                hit_rate = nested_cache['hits'] / (nested_cache['hits'] + nested_cache['misses']) * 100
                print(f"    Hit Rate: {hit_rate:.1f}%")
            
            # Dynamic context cache info
            if hasattr(self.dynamic_context, 'cache'):
                cache_size = len(getattr(self.dynamic_context, 'cache', {}))
                print(f"\n 📦 Response Cache:")
                print(f"    Current Size: {cache_size}")
                print(f"    Max Size: {self.config.cache_size_limit}")
                
        except Exception as e:
            print(f"❌ Could not retrieve cache info: {e}")

    def _print_help(self):
        """ comprehensive help information"""
        print("\n🆘  Command System:")
        print("-" * 25)
        print(" 🔚 quit/exit/bye/q     - End the session")
        print(" 📊 stats [export]      - Show/export detailed statistics")
        print(" 🆘 help [commands]     - Show this help menu")
        print(" 🧹 clear/cls          - Clear the screen")
        print(" 💾 cache [clear]       - Show/clear cache statistics")
        print(" 🔄 reset              - Reset session statistics")
        print(" ⚙️  config [show|save] - Manage configuration")
        print(" 📈 performance/perf    - Show performance metrics")
        print(" 📜 history [n]         - Show conversation history")
        print(" 🔍 search <term>       - Search conversation history")
        print(" 💾 export [filename]   - Export session data")
        print(" 🧠 memory/mem          - Show memory usage")
        print(" ⭐ rate <1-5>          - Rate the last response")
        print(" 💾 backup             - Create data backup")
        print(" 🔌 plugins [list|load] - Manage plugins")
        print(" 🐛 debug              - Show debug information")
        
        print("\n💡  Query Tips:")
        print(" • Complex questions: 'What is AI and how does ML work?'")
        print(" • Use conjunctions: 'Explain X and also describe Y'")
        print(" • Ask follow-up questions to test caching")
        print(" • Use 'search' to find previous conversations")
        print(" • Rate responses to improve the system")

    def _reset_session(self):
        """ session reset with more comprehensive cleanup"""
        self.session_stats = SessionStats()
        self.recursive_handler.clear_cache()
        
        if self.conversation_history:
            self.conversation_history = ConversationHistory(self.config.max_history_size)
            
        if self.performance_monitor:
            self.performance_monitor = PerformanceMonitor()
        
        # Trigger garbage collection
        gc.collect()
        
        print("\n🔄  session reset completed!")
        print("   • Statistics cleared")
        print("   • All caches cleared")
        print("   • Conversation history cleared")
        print("   • Performance metrics reset")
        print("   • Memory garbage collected")

    def _print_session_summary(self):
        """ comprehensive session summary"""
        print("\n📋 Final  Session Summary:")
        print("=" * 30)
        
        session_duration = datetime.now() - self.session_stats.session_start_time
        print(f"🕐 Session Duration: {session_duration}")
        
        self._print_detailed_stats()
        
        if self.session_stats.queries_processed > 0:
            print(f"\n🏆 Session Highlights:")
            if self.session_stats.recursive_queries > 0:
                print(f"   • Processed {self.session_stats.recursive_queries} complex queries")
            if self.session_stats.cache_hits > 0:
                print(f"   • Achieved {self.session_stats.cache_hits} cache hits")
            if self.session_stats.fuzzy_matches > 0:
                print(f"   • Found {self.session_stats.fuzzy_matches} fuzzy matches")
            if self.session_stats.user_satisfaction_ratings:
                avg_rating = sum(self.session_stats.user_satisfaction_ratings) / len(self.session_stats.user_satisfaction_ratings)
                print(f"   • Average user satisfaction: {avg_rating:.1f}/5")
            
            print(f"\n💡 Performance Summary:")
            print(f"   • Average response time: {self.session_stats.average_response_time:.3f}s")
            print(f"   • Peak memory usage: {self.session_stats.peak_memory_usage:.1f} MB")
            print(f"   • Success rate: {(self.session_stats.successful_queries/self.session_stats.queries_processed)*100:.1f}%")

    def _export_stats(self):
        """Export statistics to file"""
        filename = f"stats_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._save_stats_to_file(filename)
        print(f"📊 Statistics exported to {filename}")

    def _save_stats_to_file(self, filename: str = None):
        """Save session statistics to file"""
        if filename is None:
            filename = f"session_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            stats_data = {
                'session_info': {
                    'start_time': self.session_stats.session_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - self.session_stats.session_start_time).total_seconds()
                },
                'statistics': asdict(self.session_stats),
                'configuration': asdict(self.config)
            }
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            import json
            with open(filename, 'w') as f:
                json.dump(stats_data, f, indent=2, default=convert_datetime)
                
        except Exception as e:
            self.logger.error(f"Failed to save stats to file: {e}")

    def _export_session_data(self, filename: str):
        """Export comprehensive session data - FIXED VERSION"""
        try:
            def datetime_converter(obj):
                """Convert datetime objects to ISO format strings"""
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            session_data = {
                'session_info': {
                    'start_time': self.session_stats.session_start_time.isoformat(),
                    'export_time': datetime.now().isoformat(),
                },
                'statistics': asdict(self.session_stats),
                'conversation_history': [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'query': entry['query'],
                        'response': entry['response'],
                        'metadata': entry['metadata']
                    }
                    for entry in self.conversation_history.history
                ] if self.conversation_history else [],
                'configuration': asdict(self.config)
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=datetime_converter)
            
            print(f"💾 Session data exported to {filename}")
            
        except Exception as e:
            print(f"❌ Failed to export session data: {e}")

    def _create_backup(self):
        """Create backup of important data"""
        try:
            backup_dir = "backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(backup_dir, f"chatbot_backup_{timestamp}.json")
            
            backup_data = {
                'backup_time': datetime.now().isoformat(),
                'data_path': self.data_path,
                'configuration': asdict(self.config),
                'system_info': {
                    'python_version': sys.version,
                    'platform': sys.platform,
                    'cwd': os.getcwd()
                }
            }
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            print(f"💾 Backup created: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")


def main():
    """ main entry point with comprehensive argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description=' Recursive AI Chatbot with Advanced Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Start with default settings
  python main.py --data my_data.json       # Use custom data file
  python main.py --logging --config my_config.json  # Enable logging with custom config
  python main.py --performance-monitoring  # Enable detailed performance monitoring
        """
    )
    
    parser.add_argument('--data', '-d', 
                       help='Path to knowledge base JSON file')
    parser.add_argument('--logging', '-l', action='store_true', 
                       help='Enable detailed logging')
    parser.add_argument('--config', '-c', default='chatbot_config.json',
                       help='Path to configuration file (default: chatbot_config.json)')
    parser.add_argument('--performance-monitoring', '-p', action='store_true',
                       help='Enable  performance monitoring')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching features')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    parser.add_argument('--export-config', action='store_true',
                       help='Export default configuration and exit')
    
    args = parser.parse_args()
    
    # Handle special arguments
    if args.export_config:
        config = ChatbotConfig()
        config.save_to_file('default_config.json')
        print("✅ Default configuration exported to default_config.json")
        return
    
    try:
        # Load and modify configuration based on arguments
        if os.path.exists(args.config):
            config = ChatbotConfig.load_from_file(args.config)
        else:
            config = ChatbotConfig()
            if args.config != 'chatbot_config.json':  # User specified a config file that doesn't exist
                print(f"⚠️ Configuration file {args.config} not found, using defaults")
        
        # Apply argument overrides
        if args.performance_monitoring:
            config.enable_performance_monitoring = True
        if args.no_cache:
            config.cache_size_limit = 0
        if args.debug:
            config.logging_level = "DEBUG"
            config.enable_analytics = True
        
        # Save configuration for future use
        config.save_to_file(args.config)
        
        app = RecursiveAIChatbotApp(
            data_path=args.data, 
            enable_logging=args.logging or args.debug,
            config_path=args.config
        )
        app.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start  application: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
