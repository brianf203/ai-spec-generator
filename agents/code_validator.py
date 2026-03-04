"""
Code Validator and Sanitizer
Automatically detects and fixes common code generation errors:
- Syntax errors
- Indentation issues
- Import placement
- Common formatting issues
"""

import ast
import re
import textwrap
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """Represents a code validation issue"""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'syntax', 'indentation', 'import', 'formatting'
    message: str
    line: Optional[int] = None
    fix_suggestion: Optional[str] = None


class CodeValidator:
    """Validates and sanitizes LLM-generated code (not source files)"""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
    
    def validate_and_fix(self, code: str, function_name: str = None) -> Tuple[str, List[ValidationIssue], bool]:
        """
        Validate code and attempt automatic fixes.
        Returns: (fixed_code, issues, is_valid)
        """
        if not code or not code.strip():
            return "", [ValidationIssue('error', 'empty', 'Code is empty')], False
        
        self.issues = []
        fixed_code = code
        
        # Step 1: Basic cleaning
        fixed_code = self._clean_common_artifacts(fixed_code)
        
        # Step 2: Fix indentation issues
        fixed_code, indent_issues = self._fix_indentation(fixed_code)
        self.issues.extend(indent_issues)
        
        # Step 3: Fix import placement
        fixed_code, import_issues = self._fix_imports(fixed_code)
        self.issues.extend(import_issues)
        
        # Step 4: Validate syntax
        syntax_valid, syntax_issues = self._validate_syntax(fixed_code)
        self.issues.extend(syntax_issues)
        
        if not syntax_valid:
            # Step 5: Attempt syntax fixes
            fixed_code, fix_attempts = self._attempt_syntax_fixes(fixed_code, syntax_issues)
            self.issues.extend(fix_attempts)
            
            # Re-validate after fixes
            syntax_valid, remaining_issues = self._validate_syntax(fixed_code)
            self.issues.extend(remaining_issues)
        
        # Step 6: Validate structure (function/class presence)
        structure_valid, structure_issues = self._validate_structure(fixed_code, function_name)
        self.issues.extend(structure_issues)
        
        is_valid = syntax_valid and structure_valid and not any(
            i.severity == 'error' for i in self.issues
        )
        
        return fixed_code, self.issues, is_valid
    
    def _clean_common_artifacts(self, code: str) -> str:
        """Remove common LLM output artifacts"""
        # Remove markdown code blocks
        code = re.sub(r'```python\s*\n?', '', code)
        code = re.sub(r'```\s*$', '', code)
        code = re.sub(r'```\s*\n?', '', code)
        
        # Remove common prefixes
        code = re.sub(r'^Here is the code:\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^Here\'s the code:\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^The code is:\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^Here is.*?:\s*\n?', '', code, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        return code
    
    def _fix_indentation(self, code: str) -> Tuple[str, List[ValidationIssue]]:
        """Fix common indentation issues"""
        issues = []
        
        if not code.strip():
            return code, issues
        
        lines = code.split('\n')
        fixed_lines = []
        in_code = False
        
        # Find first code line
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and (stripped.startswith('def ') or stripped.startswith('class ') or 
                           stripped.startswith('import ') or stripped.startswith('from ')):
                in_code = True
            
            if in_code:
                fixed_lines.append(line)
        
        if not fixed_lines:
            # Try dedenting if everything seems indented
            try:
                dedented = textwrap.dedent(code)
                if dedented.strip():
                    # Check if dedented version is valid
                    try:
                        ast.parse(dedented)
                        issues.append(ValidationIssue(
                            'warning', 'indentation',
                            'Code was over-indented, fixed by dedenting',
                            fix_suggestion='Applied textwrap.dedent()'
                        ))
                        return dedented, issues
                    except:
                        pass
            except:
                pass
        
        fixed_code = '\n'.join(fixed_lines) if fixed_lines else code
        
        # Check for mixed indentation
        if fixed_code:
            spaces = [len(line) - len(line.lstrip()) for line in fixed_code.split('\n') 
                     if line.strip() and not line.strip().startswith('#')]
            if spaces:
                # Check if using tabs (tab = 8 spaces typically)
                has_tabs = any('\t' in line for line in fixed_code.split('\n'))
                if has_tabs:
                    fixed_code = fixed_code.expandtabs(4)
                    issues.append(ValidationIssue(
                        'warning', 'indentation',
                        'Found tabs, converted to spaces',
                        fix_suggestion='Converted tabs to 4 spaces'
                    ))
        
        return fixed_code, issues
    
    def _fix_imports(self, code: str) -> Tuple[str, List[ValidationIssue]]:
        """Fix import placement issues"""
        issues = []
        
        lines = code.split('\n')
        imports = []
        other_lines = []
        future_imports = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('from __future__'):
                future_imports.append(line)
            elif stripped.startswith('import ') or stripped.startswith('from '):
                imports.append(line)
            else:
                other_lines.append(line)
        
        # Reorder: __future__ imports first, then regular imports, then code
        if future_imports or imports:
            fixed_lines = future_imports + imports + [''] + other_lines
            fixed_code = '\n'.join(fixed_lines)
            
            if future_imports:
                issues.append(ValidationIssue(
                    'info', 'import',
                    'Reorganized imports: __future__ imports moved to top',
                    fix_suggestion='Moved __future__ imports to top'
                ))
            
            return fixed_code, issues
        
        return code, issues
    
    def _validate_syntax(self, code: str) -> Tuple[bool, List[ValidationIssue]]:
        """Validate Python syntax"""
        issues = []
        
        if not code.strip():
            issues.append(ValidationIssue('error', 'syntax', 'Code is empty'))
            return False, issues
        
        try:
            ast.parse(code)
            return True, issues
        except SyntaxError as e:
            issues.append(ValidationIssue(
                'error', 'syntax',
                f'Syntax error: {str(e)}',
                line=e.lineno,
                fix_suggestion=f'Fix syntax at line {e.lineno}: {e.msg}'
            ))
            return False, issues
        except Exception as e:
            issues.append(ValidationIssue(
                'error', 'syntax',
                f'Parse error: {str(e)}',
                fix_suggestion='Code could not be parsed'
            ))
            return False, issues
    
    def _attempt_syntax_fixes(self, code: str, syntax_issues: List[ValidationIssue]) -> Tuple[str, List[ValidationIssue]]:
        """Attempt to automatically fix syntax errors"""
        issues = []
        fixed_code = code
        
        for issue in syntax_issues:
            if issue.category != 'syntax' or issue.severity != 'error':
                continue
            
            # Try common fixes
            msg = issue.message.lower()
            
            # Fix: unexpected indent
            if 'unexpected indent' in msg or 'indentation' in msg:
                try:
                    dedented = textwrap.dedent(fixed_code)
                    try:
                        ast.parse(dedented)
                        fixed_code = dedented
                        issues.append(ValidationIssue(
                            'info', 'syntax',
                            'Fixed indentation error by dedenting',
                            line=issue.line
                        ))
                        continue
                    except:
                        pass
                except:
                    pass
            
            # Fix: missing colon
            if 'expected \':\'' in msg or 'invalid syntax' in msg:
                lines = fixed_code.split('\n')
                if issue.line and 0 < issue.line <= len(lines):
                    line_idx = issue.line - 1
                    line = lines[line_idx]
                    if line.strip() and not line.rstrip().endswith(':'):
                        # Check if it should have a colon (if, def, class, for, while, etc.)
                        stripped = line.strip()
                        if any(stripped.startswith(kw) for kw in ['if', 'def', 'class', 'for', 'while', 'elif', 'else', 'try', 'except', 'finally']):
                            if ':' not in stripped:
                                lines[line_idx] = line.rstrip() + ':'
                                fixed_code = '\n'.join(lines)
                                issues.append(ValidationIssue(
                                    'info', 'syntax',
                                    f'Added missing colon at line {issue.line}',
                                    line=issue.line
                                ))
                                continue
            
            # Fix: unclosed brackets/parentheses
            if 'unexpected eof' in msg or 'eof' in msg:
                # Try to balance brackets
                open_parens = fixed_code.count('(') - fixed_code.count(')')
                open_brackets = fixed_code.count('[') - fixed_code.count(']')
                open_braces = fixed_code.count('{') - fixed_code.count('}')
                
                if open_parens > 0:
                    fixed_code += ')' * open_parens
                    issues.append(ValidationIssue(
                        'info', 'syntax',
                        f'Added {open_parens} missing closing parenthesis',
                        fix_suggestion='Added closing parentheses'
                    ))
                elif open_brackets > 0:
                    fixed_code += ']' * open_brackets
                    issues.append(ValidationIssue(
                        'info', 'syntax',
                        f'Added {open_brackets} missing closing bracket',
                        fix_suggestion='Added closing brackets'
                    ))
                elif open_braces > 0:
                    fixed_code += '}' * open_braces
                    issues.append(ValidationIssue(
                        'info', 'syntax',
                        f'Added {open_braces} missing closing brace',
                        fix_suggestion='Added closing braces'
                    ))
        
        return fixed_code, issues
    
    def _validate_structure(self, code: str, function_name: str = None) -> Tuple[bool, List[ValidationIssue]]:
        """Validate that code has expected structure"""
        issues = []
        
        if not code.strip():
            issues.append(ValidationIssue('error', 'structure', 'Code is empty'))
            return False, issues
        
        try:
            tree = ast.parse(code)
            
            # Check for functions
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            if not functions and not classes:
                issues.append(ValidationIssue(
                    'warning', 'structure',
                    'No functions or classes found in generated code',
                    fix_suggestion='Code should contain at least one function or class'
                ))
            
            # If function_name specified, check if it exists
            if function_name:
                found = any(f.name == function_name for f in functions)
                if not found:
                    issues.append(ValidationIssue(
                        'warning', 'structure',
                        f'Expected function "{function_name}" not found',
                        fix_suggestion=f'Code should contain function named "{function_name}"'
                    ))
            
            return True, issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                'error', 'structure',
                f'Could not validate structure: {str(e)}'
            ))
            return False, issues
    
    def get_validation_summary(self) -> str:
        """Get a human-readable summary of validation issues"""
        if not self.issues:
            return "✅ Code validation passed with no issues"
        
        errors = [i for i in self.issues if i.severity == 'error']
        warnings = [i for i in self.issues if i.severity == 'warning']
        infos = [i for i in self.issues if i.severity == 'info']
        
        summary = []
        if errors:
            summary.append(f"❌ {len(errors)} error(s):")
            for err in errors[:5]:
                line_info = f" (line {err.line})" if err.line else ""
                summary.append(f"  - {err.message}{line_info}")
        
        if warnings:
            summary.append(f"⚠️  {len(warnings)} warning(s):")
            for warn in warnings[:5]:
                line_info = f" (line {warn.line})" if warn.line else ""
                summary.append(f"  - {warn.message}{line_info}")
        
        if infos:
            summary.append(f"ℹ️  {len(infos)} fix(es) applied:")
            for info in infos[:5]:
                summary.append(f"  - {info.message}")
        
        return "\n".join(summary)

