"""Read the actual channel NLLs of a live Combine object for diagnostics only."""


def channel_nlls(root, parent):
    """Return borrowed channel objects; the caller must keep parent alive.

    ROOT reflection supplies the installed class's member offset and type. No
    layout is hard-coded and no workspace, Combine source or NLL is modified.
    This uses private state and is deliberately outside the production backend.
    """
    if not isinstance(parent, root.cacheutils.CachingSimNLL):
        raise TypeError("Expected a live Combine CachingSimNLL")
    klass = root.TClass.GetClass("cacheutils::CachingSimNLL")
    member = klass.GetDataMember("pdfs_") if klass else None
    if not member or member.GetOffset() < 0:
        raise RuntimeError("Installed Combine does not expose the channel member through ROOT reflection")
    type_name = str(member.GetFullTypeName())
    if type_name.replace("std::", "").replace(" ", "") != "vector<cacheutils::CachingAddNLL*>":
        raise TypeError(f"Unsupported reflected channel container: {type_name}")
    vector = root.BindObject(root.addressof(parent)+member.GetOffset(), type_name)
    root.SetOwnership(vector, False)
    channels = []
    for item in vector:
        if not item:
            raise ValueError("Null channel NLL is unsupported by this diagnostic")
        root.SetOwnership(item, False)
        channels.append(item)
    names = [str(item.pdf().GetName()) for item in channels]
    if len(set(names)) != len(names):
        raise ValueError("Channel PDF names are not unique")
    return channels, dict(class_name="cacheutils::CachingSimNLL", member="pdfs_",
                          member_type=type_name, member_offset=int(member.GetOffset()))
