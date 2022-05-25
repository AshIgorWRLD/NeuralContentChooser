def merge(main_data, merging_data, merging_on_parameter):
    main_data = main_data.merge(merging_data, on=merging_on_parameter)
    return main_data
