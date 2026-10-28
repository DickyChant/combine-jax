"""Shared guarded compilation for statistical profile drivers."""
import time


class ProfileCompiler:
    def _compile(self, function, arguments, name, limit):
        def progress(stage):
            if self.compile_progress is not None:
                self.compile_progress(name, stage, dict(self.timings), dict(self.compiler_memory))

        def release():
            if self.release_compile_cache:
                from ..local import release_compilation_cache
                start = time.perf_counter()
                release_compilation_cache()
                self.timings['cache_release_seconds'] += time.perf_counter()-start

        progress('lowering')
        start = time.perf_counter()
        lowered = function.lower(*arguments)
        self.timings['lower_seconds'] += time.perf_counter()-start
        progress('lowered')
        release()
        progress('compiling')
        start = time.perf_counter()
        compiled = lowered.compile()
        self.timings['compile_seconds'] += time.perf_counter()-start
        memory = compiled.memory_analysis()
        if memory is None:
            raise RuntimeError('Compiler memory analysis unavailable; cannot apply the static guard')
        fields = {key: int(getattr(memory, key)) for key in ('argument_size_in_bytes',
            'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
        total = fields['argument_size_in_bytes']+fields['output_size_in_bytes']+fields['temp_size_in_bytes']-fields['alias_size_in_bytes']
        self.compiler_memory[name] = dict(**fields, total_bytes=total)
        if total > limit*2**30:
            raise MemoryError(f'{name} executable needs {total/2**30:.3f} GiB, above {limit:g} GiB guard')
        del lowered, function
        release()
        progress('compiled')
        return compiled

