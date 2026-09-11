"""A context shutdown must not hide why observation failed."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.aufgabe04.navigation.foundation.observation_node_lifecycle import (
    observation_node,
)


class ObservationNodeLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(init=Mock(), try_shutdown=Mock())
        self.node = SimpleNamespace(destroy_node=Mock())
        self.factory = Mock(return_value=self.node)

    def test_success_destroys_node_and_shuts_down_once(self):
        with observation_node(self.runtime, self.factory, "config", value=3) as node:
            self.assertIs(node, self.node)
        self.runtime.init.assert_called_once_with(args=None)
        self.factory.assert_called_once_with("config", value=3)
        self.node.destroy_node.assert_called_once_with()
        self.runtime.try_shutdown.assert_called_once_with()

    def test_already_stopped_context_uses_idempotent_shutdown(self):
        # Humble's try_shutdown handles both live and already-stopped contexts.
        self.runtime.shutdown = Mock(side_effect=RuntimeError("already shut down"))
        with observation_node(self.runtime, self.factory):
            pass
        self.runtime.shutdown.assert_not_called()
        self.runtime.try_shutdown.assert_called_once_with()

    def test_collect_exception_survives_both_cleanup_errors(self):
        original = RuntimeError("original collection failure")
        self.node.destroy_node.side_effect = ValueError("destruction failed")
        self.runtime.try_shutdown.side_effect = ValueError("shutdown failed")
        with self.assertLogs(level="ERROR"), self.assertRaises(RuntimeError) as caught:
            with observation_node(self.runtime, self.factory):
                raise original
        self.assertIs(caught.exception, original)
        self.runtime.try_shutdown.assert_called_once_with()

    def test_constructor_exception_still_shuts_down(self):
        original = ValueError("constructor failed")
        self.factory.side_effect = original
        with self.assertRaises(ValueError) as caught:
            with observation_node(self.runtime, self.factory):
                self.fail("construction must fail before yielding")
        self.assertIs(caught.exception, original)
        self.runtime.try_shutdown.assert_called_once_with()

    def test_destroy_failure_still_shuts_down_and_propagates(self):
        original = RuntimeError("destruction failed")
        self.node.destroy_node.side_effect = original
        with self.assertRaises(RuntimeError) as caught:
            with observation_node(self.runtime, self.factory):
                pass
        self.assertIs(caught.exception, original)
        self.runtime.try_shutdown.assert_called_once_with()

    def test_shutdown_failure_after_success_propagates(self):
        self.runtime.try_shutdown.side_effect = RuntimeError("shutdown failed")
        with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
            with observation_node(self.runtime, self.factory):
                pass

    def test_unrelated_outer_exception_does_not_hide_cleanup_failure(self):
        self.runtime.try_shutdown.side_effect = RuntimeError("shutdown failed")
        try:
            raise ValueError("an already handled failure")
        except ValueError:
            with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
                with observation_node(self.runtime, self.factory):
                    pass

    def test_interrupt_remains_primary(self):
        original = KeyboardInterrupt()
        self.runtime.try_shutdown.side_effect = RuntimeError("shutdown failed")
        with self.assertLogs(level="ERROR"), self.assertRaises(KeyboardInterrupt) as caught:
            with observation_node(self.runtime, self.factory):
                raise original
        self.assertIs(caught.exception, original)


if __name__ == "__main__":
    unittest.main()
