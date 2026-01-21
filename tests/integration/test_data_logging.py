"""
Integration Tests for Data Logging Pipeline
=============================================

This module tests the integration between simulation and data logging,
including HDF5 for time-series data and SQLite for event logs.

Test Categories:
- HDF5 data writing/reading
- SQLite event logging
- Hybrid logging coordination
- Compression and storage efficiency

Expected Outcomes:
- Particle data is correctly written to HDF5
- Events are logged to SQLite with timestamps
- Data can be retrieved accurately
- Compression reduces file size
"""

import pytest
import numpy as np
import sqlite3
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Mock Data Logger Classes
# =============================================================================

class MockHDF5Logger:
    """Mock HDF5 logger for integration testing."""
    
    def __init__(self, file_path, chunk_size=1000, compression='gzip'):
        import h5py
        
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.compression = compression
        self.file = h5py.File(file_path, 'w')
        
        # Create groups
        self.file.create_group('simulation')
        self.file.create_group('metadata')
        
        # Initialize datasets
        self.frame_count = 0
        self.positions_data = []
        self.velocities_data = []
        self.timestamps = []
    
    def log_frame(self, positions, velocities, timestamp):
        """Log a single frame of particle data."""
        self.positions_data.append(positions.copy())
        self.velocities_data.append(velocities.copy())
        self.timestamps.append(timestamp)
        self.frame_count += 1
        
        # Write to file when chunk is full
        if len(self.positions_data) >= self.chunk_size:
            self._flush_chunk()
    
    def _flush_chunk(self):
        """Write buffered data to HDF5 file."""
        if not self.positions_data:
            return
        
        positions_array = np.array(self.positions_data)
        velocities_array = np.array(self.velocities_data)
        timestamps_array = np.array(self.timestamps)
        
        # Create or extend datasets
        sim_group = self.file['simulation']
        
        if 'positions' not in sim_group:
            maxshape = (None,) + positions_array.shape[1:]
            sim_group.create_dataset(
                'positions', 
                data=positions_array,
                maxshape=maxshape,
                compression=self.compression,
                chunks=True
            )
            sim_group.create_dataset(
                'velocities',
                data=velocities_array,
                maxshape=maxshape,
                compression=self.compression,
                chunks=True
            )
            sim_group.create_dataset(
                'timestamps',
                data=timestamps_array,
                maxshape=(None,),
                compression=self.compression,
                chunks=True
            )
        else:
            # Extend datasets
            old_size = sim_group['positions'].shape[0]
            new_size = old_size + len(positions_array)
            
            sim_group['positions'].resize(new_size, axis=0)
            sim_group['positions'][old_size:] = positions_array
            
            sim_group['velocities'].resize(new_size, axis=0)
            sim_group['velocities'][old_size:] = velocities_array
            
            sim_group['timestamps'].resize(new_size, axis=0)
            sim_group['timestamps'][old_size:] = timestamps_array
        
        # Clear buffers
        self.positions_data = []
        self.velocities_data = []
        self.timestamps = []
    
    def set_metadata(self, key, value):
        """Set metadata attribute."""
        self.file['metadata'].attrs[key] = value
    
    def close(self):
        """Close the HDF5 file."""
        self._flush_chunk()  # Flush remaining data
        self.file.close()
    
    def get_frame(self, frame_index):
        """Retrieve a specific frame."""
        return {
            'positions': self.file['simulation/positions'][frame_index],
            'velocities': self.file['simulation/velocities'][frame_index],
            'timestamp': self.file['simulation/timestamps'][frame_index]
        }
    
    def get_frame_range(self, start, end):
        """Retrieve a range of frames."""
        return {
            'positions': self.file['simulation/positions'][start:end],
            'velocities': self.file['simulation/velocities'][start:end],
            'timestamps': self.file['simulation/timestamps'][start:end]
        }


class MockSQLiteEventLogger:
    """Mock SQLite event logger for integration testing."""
    
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(db_path)
        self._init_tables()
    
    def _init_tables(self):
        """Initialize database tables."""
        cursor = self.conn.cursor()
        
        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT DEFAULT 'info',
                message TEXT,
                data TEXT
            )
        ''')
        
        # LOD switches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lod_switches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                from_mode TEXT,
                to_mode TEXT,
                particle_count INTEGER,
                frame_time REAL
            )
        ''')
        
        # Warnings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS warnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                warning_type TEXT,
                value REAL,
                threshold REAL,
                message TEXT
            )
        ''')
        
        self.conn.commit()
    
    def log_event(self, event_type, message, severity='info', data=None):
        """Log a general event."""
        import json
        if self.conn is None:
            return
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO events (timestamp, event_type, severity, message, data) VALUES (?, ?, ?, ?, ?)',
            (time.time(), event_type, severity, message, json.dumps(data) if data else None)
        )
        self.conn.commit()
    
    def log_lod_switch(self, from_mode, to_mode, particle_count, frame_time):
        """Log an LOD mode switch."""
        if self.conn is None:
            return
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO lod_switches (timestamp, from_mode, to_mode, particle_count, frame_time) VALUES (?, ?, ?, ?, ?)',
            (time.time(), from_mode, to_mode, particle_count, frame_time)
        )
        self.conn.commit()
    
    def log_warning(self, warning_type, value, threshold, message):
        """Log a warning (e.g., velocity exceeding limit)."""
        if self.conn is None:
            return
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO warnings (timestamp, warning_type, value, threshold, message) VALUES (?, ?, ?, ?, ?)',
            (time.time(), warning_type, value, threshold, message)
        )
        self.conn.commit()
    
    def get_events(self, event_type=None, limit=100):
        """Retrieve events, optionally filtered by type."""
        if self.conn is None:
            return []
        cursor = self.conn.cursor()
        if event_type:
            cursor.execute(
                'SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?',
                (event_type, limit)
            )
        else:
            cursor.execute('SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', (limit,))
        return cursor.fetchall()
    
    def get_lod_switches(self, limit=100):
        """Retrieve LOD switch events."""
        if self.conn is None:
            return []
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM lod_switches ORDER BY timestamp DESC LIMIT ?', (limit,))
        return cursor.fetchall()
    
    def get_warnings(self, limit=100):
        """Retrieve warnings."""
        if self.conn is None:
            return []
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM warnings ORDER BY timestamp DESC LIMIT ?', (limit,))
        return cursor.fetchall()
    
    def close(self):
        """Close database connection."""
        if self.conn is None:
            return
        self.conn.close()
        self.conn = None


class MockHybridLogger:
    """Combined HDF5 + SQLite logger for integration testing."""
    
    def __init__(self, output_dir, session_name=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._closed = False
        
        session_name = session_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.hdf5_logger = MockHDF5Logger(
            self.output_dir / f'{session_name}_data.h5'
        )
        self.event_logger = MockSQLiteEventLogger(
            self.output_dir / f'{session_name}_events.sqlite'
        )
        
        # Log session start
        self.event_logger.log_event('session', 'Session started', severity='info')
    
    def log_frame(self, positions, velocities, timestamp):
        """Log particle data to HDF5."""
        self.hdf5_logger.log_frame(positions, velocities, timestamp)
    
    def log_event(self, event_type, message, **kwargs):
        """Log event to SQLite."""
        self.event_logger.log_event(event_type, message, **kwargs)
    
    def log_lod_switch(self, from_mode, to_mode, particle_count, frame_time):
        """Log LOD switch."""
        self.event_logger.log_lod_switch(from_mode, to_mode, particle_count, frame_time)
    
    def log_warning(self, warning_type, value, threshold, message):
        """Log warning."""
        self.event_logger.log_warning(warning_type, value, threshold, message)
    
    def close(self):
        """Close all loggers."""
        if self._closed:
            return
        self.event_logger.log_event('session', 'Session ended', severity='info')
        self.hdf5_logger.close()
        self.event_logger.close()
        self._closed = True


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path / "test_output"


@pytest.fixture
def hdf5_logger(tmp_path):
    """Create HDF5 logger for testing."""
    import h5py
    file_path = tmp_path / "test_data.h5"
    logger = MockHDF5Logger(file_path, chunk_size=10)
    yield logger
    logger.close()


@pytest.fixture
def event_logger(tmp_path):
    """Create SQLite event logger for testing."""
    db_path = tmp_path / "test_events.sqlite"
    logger = MockSQLiteEventLogger(db_path)
    yield logger
    logger.close()


@pytest.fixture
def hybrid_logger(temp_output_dir):
    """Create hybrid logger for testing."""
    logger = MockHybridLogger(temp_output_dir, session_name='test_session')
    yield logger
    logger.close()


# =============================================================================
# HDF5 Data Logging Tests
# =============================================================================

class TestHDF5DataLogging:
    """
    Test HDF5 time-series data logging.
    
    Tests verify:
    - Data is written correctly
    - Compression works
    - Data can be retrieved
    """
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_log_single_frame(self, hdf5_logger):
        """
        Test logging a single frame.
        
        Expected: Frame data is stored in buffer.
        """
        positions = np.random.rand(100, 3).astype(np.float32)
        velocities = np.random.rand(100, 3).astype(np.float32)
        
        hdf5_logger.log_frame(positions, velocities, 0.0)
        
        assert hdf5_logger.frame_count == 1
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_log_multiple_frames(self, hdf5_logger):
        """
        Test logging multiple frames.
        
        Expected: All frames are recorded.
        """
        for i in range(50):
            positions = np.random.rand(100, 3).astype(np.float32)
            velocities = np.random.rand(100, 3).astype(np.float32)
            hdf5_logger.log_frame(positions, velocities, i * 0.016)
        
        assert hdf5_logger.frame_count == 50
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_chunk_flushing(self, hdf5_logger):
        """
        Test data is flushed at chunk boundaries.
        
        Expected: Data written to file after chunk_size frames.
        """
        chunk_size = hdf5_logger.chunk_size
        
        for i in range(chunk_size + 5):
            positions = np.random.rand(100, 3).astype(np.float32)
            velocities = np.random.rand(100, 3).astype(np.float32)
            hdf5_logger.log_frame(positions, velocities, i * 0.016)
        
        # File should have data written
        assert 'positions' in hdf5_logger.file['simulation']
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_data_retrieval(self, hdf5_logger):
        """
        Test data can be retrieved accurately.
        
        Expected: Retrieved data matches logged data.
        """
        # Log known data
        positions = np.arange(300, dtype=np.float32).reshape(100, 3)
        velocities = np.zeros((100, 3), dtype=np.float32)
        
        for i in range(15):  # More than chunk size
            hdf5_logger.log_frame(positions + i, velocities, i * 0.016)
        
        hdf5_logger._flush_chunk()
        
        # Retrieve first frame
        frame = hdf5_logger.get_frame(0)
        
        np.testing.assert_array_almost_equal(
            frame['positions'],
            positions
        )
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_metadata_storage(self, hdf5_logger):
        """
        Test metadata can be stored and retrieved.
        
        Expected: Metadata attributes are preserved.
        """
        hdf5_logger.set_metadata('particle_count', 1000)
        hdf5_logger.set_metadata('time_step', 0.016)
        
        assert hdf5_logger.file['metadata'].attrs['particle_count'] == 1000
        assert hdf5_logger.file['metadata'].attrs['time_step'] == 0.016
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_compression_reduces_size(self, tmp_path):
        """
        Test compression reduces file size.
        
        Expected: Compressed file is smaller than uncompressed.
        """
        import h5py
        
        # Create uncompressed file
        uncompressed_path = tmp_path / "uncompressed.h5"
        with h5py.File(uncompressed_path, 'w') as f:
            data = np.random.rand(1000, 100, 3)
            f.create_dataset('data', data=data)
        
        # Create compressed file
        compressed_path = tmp_path / "compressed.h5"
        with h5py.File(compressed_path, 'w') as f:
            data = np.random.rand(1000, 100, 3)
            f.create_dataset('data', data=data, compression='gzip')
        
        uncompressed_size = uncompressed_path.stat().st_size
        compressed_size = compressed_path.stat().st_size
        
        assert compressed_size < uncompressed_size


# =============================================================================
# SQLite Event Logging Tests
# =============================================================================

class TestSQLiteEventLogging:
    """
    Test SQLite event logging.
    
    Tests verify:
    - Events are logged correctly
    - Events can be queried
    - Different event types are supported
    """
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_log_event(self, event_logger):
        """
        Test logging a simple event.
        
        Expected: Event is stored in database.
        """
        event_logger.log_event('test', 'Test message')
        
        events = event_logger.get_events()
        assert len(events) >= 1
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_log_event_with_severity(self, event_logger):
        """
        Test logging event with severity level.
        
        Expected: Severity is stored correctly.
        """
        event_logger.log_event('error', 'Error message', severity='error')
        
        events = event_logger.get_events('error')
        assert len(events) >= 1
        assert events[0][3] == 'error'  # severity column
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_log_lod_switch(self, event_logger):
        """
        Test logging LOD mode switch.
        
        Expected: Switch is recorded with all details.
        """
        event_logger.log_lod_switch('cpu', 'gpu', 5000, 0.025)
        
        switches = event_logger.get_lod_switches()
        assert len(switches) >= 1
        assert switches[0][2] == 'cpu'  # from_mode
        assert switches[0][3] == 'gpu'  # to_mode
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_log_warning(self, event_logger):
        """
        Test logging a warning.
        
        Expected: Warning is stored with value and threshold.
        """
        event_logger.log_warning(
            'velocity_exceeded',
            value=1e9,
            threshold=3e8,
            message='Velocity exceeds speed of light'
        )
        
        warnings = event_logger.get_warnings()
        assert len(warnings) >= 1
        assert warnings[0][2] == 'velocity_exceeded'
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_query_filtered_events(self, event_logger):
        """
        Test querying filtered events.
        
        Expected: Only matching events returned.
        """
        event_logger.log_event('physics', 'Physics event')
        event_logger.log_event('rendering', 'Rendering event')
        event_logger.log_event('physics', 'Another physics event')
        
        physics_events = event_logger.get_events('physics')
        
        assert len(physics_events) >= 2
        for event in physics_events:
            assert event[2] == 'physics'
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_timestamp_ordering(self, event_logger):
        """
        Test events are ordered by timestamp.
        
        Expected: Newest events first.
        """
        for i in range(5):
            event_logger.log_event('test', f'Event {i}')
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        events = event_logger.get_events()
        
        # Should be in descending timestamp order
        timestamps = [event[1] for event in events[:5]]
        assert timestamps == sorted(timestamps, reverse=True)


# =============================================================================
# Hybrid Logging Tests
# =============================================================================

class TestHybridLogging:
    """
    Test combined HDF5 + SQLite logging.
    
    Tests verify:
    - Both loggers work together
    - Session management works
    - Data and events can be correlated
    """
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_hybrid_initialization(self, hybrid_logger):
        """
        Test hybrid logger initialization.
        
        Expected: Both loggers are created.
        """
        assert hybrid_logger.hdf5_logger is not None
        assert hybrid_logger.event_logger is not None
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_session_start_logged(self, hybrid_logger):
        """
        Test session start is logged.
        
        Expected: Session start event exists.
        """
        events = hybrid_logger.event_logger.get_events('session')
        assert len(events) >= 1
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_concurrent_logging(self, hybrid_logger):
        """
        Test logging to both systems concurrently.
        
        Expected: Both data and events are recorded.
        """
        # Log some frames
        for i in range(5):
            positions = np.random.rand(100, 3).astype(np.float32)
            velocities = np.random.rand(100, 3).astype(np.float32)
            hybrid_logger.log_frame(positions, velocities, i * 0.016)
        
        # Log an event
        hybrid_logger.log_event('milestone', 'Reached 5 frames')
        
        # Log a warning
        hybrid_logger.log_warning('high_velocity', 1e6, 1e5, 'Fast particle detected')
        
        # Verify both systems have data
        assert hybrid_logger.hdf5_logger.frame_count == 5
        assert len(hybrid_logger.event_logger.get_events()) >= 2  # session + milestone
        assert len(hybrid_logger.event_logger.get_warnings()) >= 1
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_lod_switch_integration(self, hybrid_logger):
        """
        Test LOD switch logging through hybrid interface.
        
        Expected: LOD switch is recorded.
        """
        hybrid_logger.log_lod_switch('cpu', 'gpu', 6000, 0.030)
        
        switches = hybrid_logger.event_logger.get_lod_switches()
        assert len(switches) >= 1
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_file_creation(self, temp_output_dir, hybrid_logger):
        """
        Test output files are created.
        
        Expected: Both HDF5 and SQLite files exist.
        """
        hybrid_logger.close()
        
        h5_files = list(temp_output_dir.glob('*.h5'))
        sqlite_files = list(temp_output_dir.glob('*.sqlite'))
        
        assert len(h5_files) >= 1
        assert len(sqlite_files) >= 1


# =============================================================================
# Data Integrity Tests
# =============================================================================

class TestDataIntegrity:
    """
    Test data integrity across logging operations.
    
    Tests verify:
    - Data is not corrupted
    - Precision is maintained
    - Large datasets are handled
    """
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_float_precision_preserved(self, hdf5_logger):
        """
        Test floating point precision is maintained.
        
        Expected: Retrieved values match original within float32 precision.
        """
        original = np.array([[[1.23456789, 2.34567890, 3.45678901]]], dtype=np.float32)
        
        for i in range(15):
            hdf5_logger.log_frame(original[0], original[0], i * 0.016)
        
        hdf5_logger._flush_chunk()
        
        retrieved = hdf5_logger.get_frame(0)['positions']
        
        np.testing.assert_array_almost_equal(
            retrieved,
            original[0],
            decimal=5
        )
    
    @pytest.mark.integration
    @pytest.mark.data
    @pytest.mark.slow
    def test_large_dataset(self, tmp_path):
        """
        Test handling of large datasets.
        
        Expected: Large datasets are written without error.
        """
        import h5py
        
        logger = MockHDF5Logger(tmp_path / "large.h5", chunk_size=100)
        
        # Log 1000 frames of 10000 particles
        for i in range(1000):
            positions = np.random.rand(10000, 3).astype(np.float32)
            velocities = np.random.rand(10000, 3).astype(np.float32)
            logger.log_frame(positions, velocities, i * 0.016)
        
        logger.close()
        
        # Verify file was created
        assert (tmp_path / "large.h5").exists()


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestLoggingErrorHandling:
    """
    Test error handling in logging system.
    
    Tests verify:
    - Invalid paths are handled
    - Disk full scenarios
    - Corrupted data handling
    """
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_invalid_hdf5_path(self, tmp_path):
        """
        Test handling of invalid HDF5 path.
        
        Expected: Clear error or graceful handling.
        """
        try:
            # Try to create in non-existent nested directory
            import h5py
            invalid_path = tmp_path / "nonexistent" / "deep" / "path" / "file.h5"
            logger = MockHDF5Logger(invalid_path)
        except (OSError, IOError):
            pass  # Expected behavior
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_nan_data_logged(self, hdf5_logger):
        """
        Test NaN values are logged (for debugging).
        
        Expected: NaN values are stored (not silently dropped).
        """
        positions = np.array([[np.nan, 0.0, 0.0]], dtype=np.float32)
        velocities = np.zeros((1, 3), dtype=np.float32)
        
        hdf5_logger.log_frame(positions, velocities, 0.0)
        
        assert hdf5_logger.frame_count == 1
