import yaml
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any

class PromptGenerator:
    def __init__(self, templates_dir: str = "src/prompts"):
        self.jinja_env = Environment(
            loader=FileSystemLoader(templates_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def load_person_config(self, config_path: str) -> Dict[str, Any]:
        """加载目标人物配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def prepare_template_variables(self, person_config: Dict[str, Any]) -> Dict[str, Any]:
        """准备模板变量"""
        target_person = person_config['target_person']
        all_names = [target_person['main_name']] + target_person.get('aliases', [])

        # 生成带冒号的所有别名
        all_aliases_with_colon = '、'.join([f"{name}：" for name in all_names])

        return {
            'target_person': target_person,
            'questioner_types': person_config.get('questioner_types', []),
            'domain_context': person_config.get('domain_context', ''),
            'sample_examples': person_config.get('sample_examples', []),
            'output_format': person_config.get('output_format', {
                'question_field': 'question',
                'answer_field': 'answer'
            }),
            'all_aliases_with_colon': all_aliases_with_colon
        }

    def generate_compact_prompt(self, person_config: Dict[str, Any]) -> str:
        """生成精简版prompt"""
        template_vars = self.prepare_template_variables(person_config)
        template = self.jinja_env.get_template('compact_prompt.j2')
        return template.render(**template_vars)

    def generate_full_prompt(self, person_config: Dict[str, Any]) -> str:
        """生成完整版prompt"""
        template_vars = self.prepare_template_variables(person_config)
        template = self.jinja_env.get_template('full_prompt.j2')
        return template.render(**template_vars)

