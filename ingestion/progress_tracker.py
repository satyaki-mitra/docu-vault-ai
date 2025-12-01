# DEPENDENCIES
import time
import uuid
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class TaskStatus(str, Enum):
    """
    Task status enumeration
    """
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


@dataclass
class TaskProgress:
    """
    Progress tracking data structure
    """
    task_id                     : str
    description                 : str
    status                      : TaskStatus
    total_items                 : int
    processed_items             : int
    current_item                : Optional[str]
    current_status              : str
    start_time                  : datetime
    end_time                    : Optional[datetime]
    progress_percent            : float
    estimated_seconds_remaining : Optional[float]
    metadata                    : Dict[str, Any]


class ProgressTracker:
    """
    Comprehensive progress tracking for long-running operations: Provides real-time progress monitoring and status updates
    """
    def __init__(self, max_completed_tasks: int = 100):
        """
        Initialize progress tracker
        
        Arguments:
        ----------
            max_completed_tasks { int } : Maximum number of completed tasks to keep in history
        """
        self.logger                                         = logger
        self.max_completed_tasks                            = max_completed_tasks
        
        # Task storage
        self.active_tasks         : Dict[str, TaskProgress] = dict()
        self.completed_tasks      : List[TaskProgress]      = list()
        self.failed_tasks         : List[TaskProgress]      = list()
        
        self.logger.info("Initialized ProgressTracker")
    

    def start_task(self, total_items: int, description: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking a new task
        
        Arguments:
        ----------
            total_items { int }  : Total number of items to process

            description { str }  : Task description
            
            metadata    { dict } : Additional task metadata
        
        Returns:
        --------
               { str }           : Generated task ID
        """
        task_id                    = self._generate_task_id()
        
        task_progress              = TaskProgress(task_id                     = task_id,
                                                  description                 = description,
                                                  status                      = TaskStatus.RUNNING,
                                                  total_items                 = total_items,
                                                  processed_items             = 0,
                                                  current_item                = None,
                                                  current_status              = "Starting...",
                                                  start_time                  = datetime.now(),
                                                  end_time                    = None,
                                                  progress_percent            = 0.0,
                                                  estimated_seconds_remaining = None,
                                                  metadata                    = metadata or {},
                                                 )
        
        self.active_tasks[task_id] = task_progress
        
        self.logger.info(f"Started task {task_id}: {description} ({total_items} items)")
        
        return task_id
    

    def update_task(self, task_id: str, processed_items: Optional[int] = None, current_item: Optional[str] = None, 
                    current_status: Optional[str] = None, metadata_update: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update task progress
        
        Arguments:
        ----------
            task_id         { str }  : Task ID to update

            processed_items { int }  : Number of items processed
            
            current_item    { str }  : Current item being processed
            
            current_status  { str }  : Current status message
            
            metadata_update { dict } : Metadata updates
        
        Returns:
        --------
                 { bool }            : True if update successful
        """
        if task_id not in self.active_tasks:
            self.logger.warning(f"Task {task_id} not found in active tasks")
            return False
        
        task = self.active_tasks[task_id]
        
        # Update fields
        if processed_items is not None:
            task.processed_items = processed_items
        
        if current_item is not None:
            task.current_item = current_item
        
        if current_status is not None:
            task.current_status = current_status
        
        if metadata_update:
            task.metadata.update(metadata_update)
        
        # Calculate progress
        if (task.total_items > 0):
            task.progress_percent = (task.processed_items / task.total_items) * 100.0
            
            # Estimate remaining time
            if (task.processed_items > 0):
                elapsed_seconds  = (datetime.now() - task.start_time).total_seconds()
                items_per_second = task.processed_items / elapsed_seconds
                
                if (items_per_second > 0):
                    remaining_items                  = task.total_items - task.processed_items
                    task.estimated_seconds_remaining = remaining_items / items_per_second
        
        # Ensure progress doesn't exceed 100%
        if (task.progress_percent > 100.0):
            task.progress_percent = 100.0
        
        self.logger.debug(f"Updated task {task_id}: {task.progress_percent:.1f}% complete")
        
        return True
    

    def complete_task(self, task_id: str, final_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark task as completed
        
        Arguments:
        ----------
            task_id         { str }  : Task ID to complete

            final_metadata  { dict } : Final metadata updates
        
        Returns:
        --------
                 { bool }            : True if completion successful
        """
        if task_id not in self.active_tasks:
            self.logger.warning(f"Task {task_id} not found for completion")
            return False
        
        task                             = self.active_tasks[task_id]
        
        # Update task
        task.status                      = TaskStatus.COMPLETED
        task.end_time                    = datetime.now()
        task.processed_items             = task.total_items  # Ensure 100% completion
        task.progress_percent            = 100.0
        task.estimated_seconds_remaining = 0.0
        task.current_status              = "Completed"
        
        if final_metadata:
            task.metadata.update(final_metadata)
        
        # Move to completed tasks
        self.completed_tasks.append(task)
        del self.active_tasks[task_id]
        
        # Maintain history size
        if (len(self.completed_tasks) > self.max_completed_tasks):
            self.completed_tasks = self.completed_tasks[-self.max_completed_tasks:]
        
        total_time = (task.end_time - task.start_time).total_seconds()
        
        self.logger.info(f"Completed task {task_id}: {task.description} in {total_time:.2f}s")
        
        return True
    

    def fail_task(self, task_id: str, error_message: str, error_details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark task as failed
        
        Arguments:
        ----------
            task_id        { str }  : Task ID to mark as failed

            error_message  { str }  : Error message
            
            error_details  { dict } : Additional error details
        
        Returns:
        --------
                 { bool }           : True if failure marking successful
        """
        if (task_id not in self.active_tasks):
            self.logger.warning(f"Task {task_id} not found for failure marking")
            return False
        
        task                           = self.active_tasks[task_id]
        
        # Update task
        task.status                    = TaskStatus.FAILED
        task.end_time                  = datetime.now()
        task.current_status            = f"Failed: {error_message}"
        
        if error_details:
            task.metadata["error_details"] = error_details
        
        task.metadata["error_message"] = error_message
        
        # Move to failed tasks
        self.failed_tasks.append(task)
        del self.active_tasks[task_id]
        
        total_time                     = (task.end_time - task.start_time).total_seconds()
        
        self.logger.error(f"Task {task_id} failed after {total_time:.2f}s: {error_message}")
        
        return True
    

    def cancel_task(self, task_id: str, reason: str = "User cancelled") -> bool:
        """
        Cancel a running task
        
        Arguments:
        ----------
            task_id { str } : Task ID to cancel

            reason  { str } : Cancellation reason
        
        Returns:
        --------
               { bool }     : True if cancellation successful
        """
        if task_id not in self.active_tasks:
            self.logger.warning(f"Task {task_id} not found for cancellation")
            return False
        
        task                                 = self.active_tasks[task_id]
        
        # Update task
        task.status                          = TaskStatus.CANCELLED
        task.end_time                        = datetime.now()
        task.current_status                  = f"Cancelled: {reason}"
        task.metadata["cancellation_reason"] = reason
        
        # Move to completed tasks (as cancelled)
        self.completed_tasks.append(task)
        del self.active_tasks[task_id]
        
        total_time                           = (task.end_time - task.start_time).total_seconds()
        
        self.logger.info(f"Cancelled task {task_id} after {total_time:.2f}s: {reason}")
        
        return True
    

    def get_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """
        Get current progress for a task
        
        Arguments:
        ----------
            task_id { str }     : Task ID
        
        Returns:
        --------
               { TaskProgress } : Task progress or None if not found
        """
        # Check active tasks first
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # Check completed tasks
        for task in self.completed_tasks:
            if (task.task_id == task_id):
                return task
        
        # Check failed tasks
        for task in self.failed_tasks:
            if (task.task_id == task_id):
                return task
        
        return None
    

    def get_all_active_tasks(self) -> List[TaskProgress]:
        """
        Get all active tasks
        
        Returns:
        --------
            { list }    : List of active task progresses
        """
        return list(self.active_tasks.values())
    

    def get_active_task_count(self) -> int:
        """
        Get number of active tasks
        
        Returns:
        --------
            { int }    : Number of active tasks
        """
        return len(self.active_tasks)
    

    def get_recent_completed_tasks(self, limit: int = 10) -> List[TaskProgress]:
        """
        Get recently completed tasks
        
        Arguments:
        ----------
            limit { int } : Maximum number of tasks to return
        
        Returns:
        --------
               { list }   : List of completed tasks (newest first)
        """
        # Return newest first
        return self.completed_tasks[-limit:][::-1]  
    

    def get_task_statistics(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a task
        
        Arguments:
        ----------
            task_id { str } : Task ID
        
        Returns:
        --------
               { dict }     : Task statistics or None
        """
        task = self.get_task_progress(task_id)
        
        if not task:
            return None
        
        stats = {"task_id"          : task.task_id,
                 "description"      : task.description,
                 "status"           : task.status.value,
                 "progress_percent" : task.progress_percent,
                 "processed_items"  : task.processed_items,
                 "total_items"      : task.total_items,
                 "current_item"     : task.current_item,
                 "current_status"   : task.current_status,
                 "start_time"       : task.start_time.isoformat(),
                 "metadata"         : task.metadata,
                }
        
        if task.end_time:
            total_seconds               = (task.end_time - task.start_time).total_seconds()
            stats["total_time_seconds"] = total_seconds
            stats["end_time"]           = task.end_time.isoformat()
        
        if task.estimated_seconds_remaining:
            stats["estimated_seconds_remaining"] = task.estimated_seconds_remaining
        
        return stats
    

    def get_global_statistics(self) -> Dict[str, Any]:
        """
        Get global progress statistics
        
        Returns:
        --------
            { dict }    : Global statistics
        """
        total_completed     = len(self.completed_tasks)
        total_failed        = len(self.failed_tasks)
        total_tasks         = total_completed + total_failed + len(self.active_tasks)
        
        # Calculate average completion time
        avg_completion_time = 0.0
        
        if (total_completed > 0):
            total_time          = sum((task.end_time - task.start_time).total_seconds() for task in self.completed_tasks if task.end_time)
            avg_completion_time = total_time / total_completed
        
        return {"active_tasks"                : len(self.active_tasks),
                "completed_tasks"             : total_completed,
                "failed_tasks"                : total_failed,
                "total_tasks"                 : total_tasks,
                "success_rate"                : (total_completed / total_tasks * 100) if total_tasks > 0 else 0,
                "avg_completion_time_seconds" : avg_completion_time,
                "max_completed_tasks"         : self.max_completed_tasks,
               }
    

    def cleanup_completed_tasks(self, older_than_hours: int = 24):
        """
        Clean up old completed tasks
        
        Arguments:
        ----------
            older_than_hours { int } : Remove tasks older than this many hours
        """
        cutoff_time          = datetime.now().timestamp() - (older_than_hours * 3600)
        initial_count        = len(self.completed_tasks)
        
        self.completed_tasks = [task for task in self.completed_tasks if task.end_time and task.end_time.timestamp() > cutoff_time]
        
        removed_count        = initial_count - len(self.completed_tasks)
        
        if (removed_count > 0):
            self.logger.info(f"Cleaned up {removed_count} completed tasks older than {older_than_hours} hours")
    

    @staticmethod
    def _generate_task_id() -> str:
        """
        Generate unique task ID
        
        Returns:
        --------
            { str }    : Generated task ID
        """
        return f"task_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    

    def __del__(self):
        """
        Cleanup on destruction
        """
        try:
            self.cleanup_completed_tasks()

        except Exception:
            # Ignore cleanup errors during destruction
            pass  


# Global progress tracker instance
_progress_tracker = None


def get_progress_tracker() -> ProgressTracker:
    """
    Get global progress tracker instance
    
    Returns:
    --------
        { ProgressTracker } : ProgressTracker instance
    """
    global _progress_tracker

    if _progress_tracker is None:
        _progress_tracker = ProgressTracker()
    
    return _progress_tracker


def start_progress_task(total_items: int, description: str, **kwargs) -> str:
    """
    Convenience function to start a progress task
    
    Arguments:
    ----------
        total_items { int } : Total items
        
        description { str } : Task description
        
        **kwargs            : Additional arguments
    
    Returns:
    --------
               { str }      : Task ID
    """
    tracker = get_progress_tracker()

    return tracker.start_task(total_items, description, **kwargs)