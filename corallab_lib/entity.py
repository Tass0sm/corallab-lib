from .backend_manager import backend_manager as corallab_lib_backend_manager


class BaseEntity:
    def __init__(
            self,
            entity_impl_name,
            backend_manager,
            *args,
            backend=None,
            from_impl=None,
            **kwargs
    ):
        EntityImpl = backend_manager.get_backend_attr(
            entity_impl_name,
            backend=backend
        )

        if from_impl:
            self.entity_impl = EntityImpl.from_impl(from_impl, *args, **kwargs)
        else:
            self.entity_impl = EntityImpl(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.entity_impl, name):
            return getattr(self.entity_impl, name)
        else:
            # Default behaviour
            raise AttributeError


class Entity(BaseEntity):
    def __init__(
            self,
            entity_impl_name,
            *args,
            **kwargs,
    ):
        super().__init__(
            entity_impl_name,
            corallab_lib_backend_manager,
            *args,
            **kwargs
        )
