"""
Tests for the tree tool functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import subprocess
from src.gemini_cli.tools.tree_tool import TreeTool

class TestTreeTool(unittest.TestCase):
    def setUp(self):
        self.tree_tool = TreeTool()

    @patch('subprocess.run')
    def test_execute_with_string_depth(self, mock_run):
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Directory structure"
        mock_run.return_value = mock_process

        # Test with string depth
        result = self.tree_tool.execute(depth="3")
        
        # Verify correct command construction
        mock_run.assert_called_with(
            ['tree', '-L', '3'],
            capture_output=True,
            text=True,
            check=False,
            timeout=15
        )
        self.assertEqual(result, "Directory structure")

    @patch('subprocess.run')
    def test_execute_with_invalid_depth(self, mock_run):
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Directory structure"
        mock_run.return_value = mock_process

        # Test with invalid depth
        result = self.tree_tool.execute(depth="invalid")
        
        # Should use default depth (3)
        mock_run.assert_called_with(
            ['tree', '-L', '3'],
            capture_output=True,
            text=True,
            check=False,
            timeout=15
        )

    @patch('subprocess.run')
    def test_execute_with_path(self, mock_run):
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Directory structure"
        mock_run.return_value = mock_process

        # Test with path
        result = self.tree_tool.execute(path="test_dir", depth=2)
        
        # Verify correct command construction with path
        mock_run.assert_called_with(
            ['tree', '-L', '2', 'test_dir'],
            capture_output=True,
            text=True,
            check=False,
            timeout=15
        )

if __name__ == '__main__':
    unittest.main()