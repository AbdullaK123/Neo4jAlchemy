# tests/orm/test_engine.py

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock, ANY
import re
from neo4j import AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable # For simulating connection errors

# Adjust the import path based on your project structure
from neo4jalchemy.orm.engine import GraphEngine

# --- Constants for testing ---
TEST_URI = "bolt://mockenginehost:7687"
TEST_AUTH = ("testengineuser", "testenginepass")
TEST_DEFAULT_DB = "enginedb"

# --- Fixtures ---

@pytest_asyncio.fixture
async def engine_instance(request) -> GraphEngine:
    """
    Provides a GraphEngine instance for testing.
    It's not connected by default.
    Allows parameterization for different engine configurations if needed later.
    """
    uri = getattr(request, "param", {}).get("uri", TEST_URI)
    auth = getattr(request, "param", {}).get("auth", TEST_AUTH)
    database = getattr(request, "param", {}).get("database", TEST_DEFAULT_DB)
    driver_config = getattr(request, "param", {}).get("driver_config", None)
    
    engine = GraphEngine(uri=uri, auth=auth, database=database, driver_config=driver_config)
    # Ensure it's clean before test
    assert not engine.connected
    assert engine._driver is None
    
    yield engine # Provide the engine to the test

    # Teardown: ensure the engine is closed if it was connected during a test
    if engine.connected:
        await engine.close()
    # Double check it's cleaned up
    assert engine._driver is None 
    assert not engine.connected


@pytest_asyncio.fixture
def mock_neo4j_driver_factory():
    """
    Provides a mock for neo4j.AsyncGraphDatabase.driver.
    The mock_driver_instance it returns can be further configured in tests.
    """
    mock_driver_instance = AsyncMock(spec=AsyncDriver)
    mock_driver_instance.verify_connectivity = AsyncMock()
    mock_driver_instance.close = AsyncMock()
    mock_session_instance = AsyncMock(spec=AsyncSession)
    mock_driver_instance.session = MagicMock(return_value=mock_session_instance)

    with patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver_instance) as mock_factory:
        yield mock_factory, mock_driver_instance, mock_session_instance


# --- Test Cases ---

class TestGraphEngineInitialization:
    def test_engine_initialization_defaults(self):
        engine = GraphEngine(uri=TEST_URI, auth=TEST_AUTH)
        assert engine.uri == TEST_URI
        assert engine.auth == TEST_AUTH
        assert engine.default_database == "neo4j" # Default from GraphEngine __init__
        assert not engine.connected
        assert engine._driver is None
        assert "Neo4jAlchemyEngine/0.1.0" in engine.driver_config["user_agent"]

    def test_engine_initialization_custom(self):
        custom_db = "mycustomdb"
        custom_config = {"user_agent": "MyTestApp", "max_pool_size": 10}
        engine = GraphEngine(
            uri=TEST_URI,
            auth=TEST_AUTH,
            database=custom_db,
            driver_config=custom_config
        )
        assert engine.default_database == custom_db
        assert engine.driver_config["user_agent"] == "MyTestApp"
        assert engine.driver_config["max_pool_size"] == 10
        # Ensure defaults are still there if not overridden
        assert engine.driver_config["keep_alive"] is True


@pytest.mark.asyncio
class TestGraphEngineConnect:
    async def test_connect_successful(self, engine_instance: GraphEngine, mock_neo4j_driver_factory, capsys):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory

        await engine_instance.connect()

        mock_factory.assert_called_once_with(
            engine_instance.uri,
            auth=engine_instance.auth,
            **engine_instance.driver_config
        )
        mock_driver.verify_connectivity.assert_awaited_once()
        assert engine_instance.connected is True
        assert engine_instance._driver is mock_driver
        
        captured = capsys.readouterr()
        assert f"GraphEngine: Successfully connected to {engine_instance.uri}" in captured.out

    async def test_connect_idempotent(self, engine_instance: GraphEngine, mock_neo4j_driver_factory, capsys):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory

        await engine_instance.connect() # First call
        assert engine_instance.connected is True
        first_driver_instance = engine_instance._driver

        await engine_instance.connect() # Second call
        
        mock_factory.assert_called_once() # Factory should still only be called once
        mock_driver.verify_connectivity.assert_awaited_once() # verify_connectivity also only once
        assert engine_instance.connected is True
        assert engine_instance._driver is first_driver_instance # Should be the same driver instance
        
        # capsys.readouterr() # First connect message is captured
        # captured = capsys.readouterr() # Second call should ideally not print "Connecting..." again
        # The print statement for "Already connected" is commented out in GraphEngine, so no second message.

    async def test_connect_failure_driver_creation(self, engine_instance: GraphEngine, mock_neo4j_driver_factory, capsys):
        mock_factory, _, _ = mock_neo4j_driver_factory
        mock_factory.side_effect = ServiceUnavailable("Cannot create driver")

        with pytest.raises(ConnectionError, match=re.escape(f"Failed to connect to Neo4j at {engine_instance.uri}: Cannot create driver")):
            await engine_instance.connect()

        assert engine_instance.connected is False
        assert engine_instance._driver is None
        captured = capsys.readouterr()
        assert f"GraphEngine: Connection to {engine_instance.uri} failed: Cannot create driver" in captured.out

    async def test_connect_failure_verify_connectivity(self, engine_instance: GraphEngine, mock_neo4j_driver_factory, capsys):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory
        mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Verification failed")

        with pytest.raises(ConnectionError, match=re.escape(f"Failed to connect to Neo4j at {engine_instance.uri}: Verification failed")):
            await engine_instance.connect()
        
        assert engine_instance.connected is False
        # _driver might be assigned before verify_connectivity fails, so we check it's None after failure handling
        assert engine_instance._driver is None 
        captured = capsys.readouterr()
        assert f"GraphEngine: Connection to {engine_instance.uri} failed: Verification failed" in captured.out

    async def test_connect_concurrent_calls(self, engine_instance: GraphEngine, mock_neo4j_driver_factory):
        """Test that concurrent connect calls are handled correctly by the lock."""
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory
        
        # Simulate multiple concurrent calls to connect
        connect_tasks = [engine_instance.connect() for _ in range(5)]
        await asyncio.gather(*connect_tasks)

        # The factory and verify_connectivity should still only be called once
        # due to the lock and idempotency check.
        mock_factory.assert_called_once()
        mock_driver.verify_connectivity.assert_awaited_once()
        assert engine_instance.connected is True


@pytest.mark.asyncio
class TestGraphEngineClose:
    async def test_close_active_connection(self, engine_instance: GraphEngine, mock_neo4j_driver_factory, capsys):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory
        await engine_instance.connect() # Establish connection
        assert engine_instance.connected is True

        await engine_instance.close()

        mock_driver.close.assert_awaited_once()
        assert engine_instance.connected is False
        assert engine_instance._driver is None
        captured = capsys.readouterr()
        assert f"GraphEngine: Connection to {engine_instance.uri} closed." in captured.out

    async def test_close_when_not_connected(self, engine_instance: GraphEngine, capsys):
        assert not engine_instance.connected
        await engine_instance.close() # Should not raise error
        
        captured = capsys.readouterr()
        # Check that it doesn't print "Closing connection..." if there was nothing to close
        # The current implementation prints "No active or initialized driver..." if _driver is None
        # If _driver exists but _is_connected is False, it prints "Driver ... exists but was not fully connected"
        # For a pristine engine, _driver is None, so no "Closing..." message.
        assert "Closing connection" not in captured.out 
        assert engine_instance.connected is False
        assert engine_instance._driver is None

    async def test_close_when_driver_exists_but_not_fully_connected(self, engine_instance: GraphEngine, mock_neo4j_driver_factory, capsys):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory
        
        # Simulate a state where _driver is assigned but _is_connected is false
        # (e.g., verify_connectivity failed after driver assignment but before full error handling reset _driver)
        # To achieve this for testing, we'll patch connect to simulate partial failure
        async def mock_connect_partial_failure(self_engine):
            async with self_engine._connection_lock:
                if self_engine._is_connected and self_engine._driver: return
                self_engine._driver = mock_driver # Assign driver
                # Skip verify_connectivity or simulate its failure leading to _is_connected=False
                self_engine._is_connected = False 
                # raise ConnectionError("Simulated partial connect fail") # Don't raise to test close path

        with patch.object(GraphEngine, 'connect', new=mock_connect_partial_failure):
            await engine_instance.connect() # This will run our mock_connect_partial_failure

        assert engine_instance._driver is mock_driver # Driver is assigned
        assert not engine_instance.connected        # But not connected

        await engine_instance.close()

        mock_driver.close.assert_awaited_once() # Should still attempt to close the driver
        assert engine_instance._driver is None
        assert not engine_instance.connected
        captured = capsys.readouterr()
        assert f"GraphEngine: Driver for {engine_instance.uri} exists but was not fully connected. Attempting close." in captured.out

    async def test_close_idempotent(self, engine_instance: GraphEngine, mock_neo4j_driver_factory):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory
        await engine_instance.connect()
        
        await engine_instance.close() # First close
        await engine_instance.close() # Second close

        mock_driver.close.assert_awaited_once() # Should only be called once
        assert not engine_instance.connected

    async def test_close_concurrent_calls(self, engine_instance: GraphEngine, mock_neo4j_driver_factory):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory
        await engine_instance.connect()
        
        close_tasks = [engine_instance.close() for _ in range(5)]
        await asyncio.gather(*close_tasks)

        mock_driver.close.assert_awaited_once()
        assert not engine_instance.connected


@pytest.mark.asyncio
class TestGraphEngineGetSession:
    async def test_get_session_when_connected(self, engine_instance: GraphEngine, mock_neo4j_driver_factory):
        mock_factory, mock_driver, mock_session = mock_neo4j_driver_factory
        await engine_instance.connect()

        # Test with engine's default database
        session1 = engine_instance.get_session()
        assert session1 is mock_session
        mock_driver.session.assert_called_with(database=engine_instance.default_database)

        # Test with a specific database override
        custom_db = "anotherdb"
        session2 = engine_instance.get_session(database=custom_db)
        assert session2 is mock_session
        mock_driver.session.assert_called_with(database=custom_db)

    async def test_get_session_when_not_connected(self, engine_instance: GraphEngine):
        assert not engine_instance.connected
        with pytest.raises(ConnectionError, match=re.escape(f"GraphEngine for {engine_instance.uri} is not connected. Call `await engine.connect()` first.")):
            engine_instance.get_session()


@pytest.mark.asyncio
class TestGraphEngineDriverProperty:
    async def test_driver_property_when_connected(self, engine_instance: GraphEngine, mock_neo4j_driver_factory):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory
        await engine_instance.connect()
        assert engine_instance.driver is mock_driver

    async def test_driver_property_when_not_connected(self, engine_instance: GraphEngine):
        with pytest.raises(ConnectionError, match=re.escape(f"GraphEngine for {engine_instance.uri} is not connected. Call `await engine.connect()` first.")):
            _ = engine_instance.driver


@pytest.mark.asyncio
class TestGraphEngineConnectedProperty:
    async def test_connected_property_states(self, engine_instance: GraphEngine, mock_neo4j_driver_factory):
        mock_factory, _, _ = mock_neo4j_driver_factory
        assert not engine_instance.connected # Initial state

        await engine_instance.connect()
        assert engine_instance.connected # After successful connect

        await engine_instance.close()
        assert not engine_instance.connected # After close

    async def test_connected_property_on_connect_failure(self, engine_instance: GraphEngine, mock_neo4j_driver_factory):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory
        mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Failed")
        
        with pytest.raises(ConnectionError):
            await engine_instance.connect()
        assert not engine_instance.connected


@pytest.mark.asyncio
class TestGraphEngineContextManager:
    async def test_engine_as_async_context_manager_success(self, engine_instance: GraphEngine, mock_neo4j_driver_factory, capsys):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory

        async with engine_instance as connected_engine:
            assert connected_engine is engine_instance
            assert engine_instance.connected is True
            mock_factory.assert_called_once()
            mock_driver.verify_connectivity.assert_awaited_once()
            # Check print statement from connect
            captured_on_enter = capsys.readouterr()
            assert f"GraphEngine: Successfully connected to {engine_instance.uri}" in captured_on_enter.out
        
        assert engine_instance.connected is False # Should be closed on exit
        mock_driver.close.assert_awaited_once()
        captured_on_exit = capsys.readouterr()
        assert f"GraphEngine: Connection to {engine_instance.uri} closed." in captured_on_exit.out


    async def test_engine_as_async_context_manager_connect_failure(self, engine_instance: GraphEngine, mock_neo4j_driver_factory):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory
        mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Context connect failed")

        with pytest.raises(ConnectionError, match=re.escape(f"Failed to connect to Neo4j at {engine_instance.uri}: Context connect failed")):
            async with engine_instance:
                pytest.fail("Should not reach inside context if connect fails") # Should not execute
        
        assert not engine_instance.connected
        mock_driver.close.assert_not_awaited() # Close shouldn't be called if connect failed to establish driver

    async def test_engine_as_async_context_manager_error_inside_with_block(self, engine_instance: GraphEngine, mock_neo4j_driver_factory):
        mock_factory, mock_driver, _ = mock_neo4j_driver_factory
        
        class CustomTestError(Exception): pass

        with pytest.raises(CustomTestError):
            async with engine_instance:
                assert engine_instance.connected is True
                raise CustomTestError("Error inside with block")
        
        # Engine should still be closed even if an error occurred inside the 'with' block
        assert not engine_instance.connected
        mock_driver.close.assert_awaited_once()

