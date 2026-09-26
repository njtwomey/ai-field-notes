"""Base class for every command-line entry point in the repository.

Commands are pydantic-settings models: fields are options, ``Field(description=...)`` is the ``--help`` text and the
class docstring is the command summary. Every field must carry a description so that each CLI documents itself.
"""

from typing import Any, ClassVar, Self

from pydantic_settings import BaseSettings, CliApp, PydanticBaseSettingsSource, SettingsConfigDict


class Command(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_enforce_required=True,
        cli_avoid_json=True,
        cli_hide_none_type=True,
        cli_use_class_docs_for_groups=True,
        extra="forbid",
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        undocumented = [name for name, field in cls.model_fields.items() if not field.description]
        if undocumented:
            raise TypeError(f"{cls.__name__}: every CLI field needs Field(description=...); missing: {undocumented}")
        if not cls.__doc__ or cls.__doc__ == Command.__doc__:
            raise TypeError(f"{cls.__name__}: a CLI command needs a docstring; it becomes the --help summary")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Only CLI arguments and explicit keyword arguments. Environment variables never change results.
        return (init_settings,)

    def cli_cmd(self) -> None:
        """Run the command. Subclasses with subcommands call ``CliApp.run_subcommand(self)``."""
        raise NotImplementedError

    @classmethod
    def main(cls, args: list[str] | None = None, **overrides: Any) -> Self:
        """Parse ``args`` (default: ``sys.argv[1:]``) and run ``cli_cmd``."""
        return CliApp.run(cls, cli_args=args, **overrides)
