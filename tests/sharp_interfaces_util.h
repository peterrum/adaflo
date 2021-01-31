


namespace dealii
{
  namespace VectorTools
  {
    template <int dim, int spacedim, typename VectorType>
    void
    VectorTools::get_position_vector(const DoFHandler<dim, spacedim> &dof_handler_dim,
                                     VectorType &                     euler_vector,
                                     const Mapping<dim, spacedim> &   mapping)
    {
      FEValues<dim, spacedim> fe_eval(
        mapping,
        dof_handler_dim.get_fe(),
        Quadrature<dim>(dof_handler_dim.get_fe().get_unit_support_points()),
        update_quadrature_points);

      Vector<double> temp;

      for (const auto &cell : dof_handler_dim.active_cell_iterators())
        {
          fe_eval.reinit(cell);

          temp.reinit(fe_eval.dofs_per_cell);

          for (const auto q : fe_eval.quadrature_point_indices())
            {
              const auto point = fe_eval.quadrature_point(q);

              const unsigned int comp =
                dof_handler_dim.get_fe().system_to_component_index(q).first;

              temp[q] = point[comp];
            }

          cell->set_dof_values(temp, euler_vector);
        }
    }
  } // namespace VectorTools

  namespace GridTools
  {
    template <int dim, int spacedim, typename VectorType>
    void
    within(const MappingFEField<dim, spacedim> &mapping,
           const DoFHandler<dim, spacedim> &    dof_handler,
           const Mapping<spacedim> &            background_mapping,
           const DoFHandler<spacedim> &         background_dof_handler,
           VectorType &                         force_vector_sharp_interface)
    {
      typedef boost::geometry::model::d2::point_xy<double> point_type;
      typedef boost::geometry::model::polygon<point_type>  polygon_type;

      std::vector<boost::geometry::model::d2::point_xy<double>> points;
      {
        FEValues<dim, spacedim> fe_eval(mapping,
                                        dof_handler.get_fe(),
                                        QGauss<dim>(dof_handler.get_fe().degree + 1),
                                        update_quadrature_points);

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            fe_eval.reinit(cell);

            for (const auto q : fe_eval.quadrature_point_indices())
              {
                const auto point = fe_eval.quadrature_point(q);
                points.emplace_back(point[0], point[1]);
              }
          }
      }

      polygon_type poly;
      boost::geometry::assign_points(poly, points);

      {
        FEValues<spacedim> fe_eval(
          background_mapping,
          background_dof_handler.get_fe(),
          background_dof_handler.get_fe().get_unit_support_points(),
          update_quadrature_points);

        for (const auto &cell : background_dof_handler.active_cell_iterators())
          {
            fe_eval.reinit(cell);

            Vector<double> vec(fe_eval.n_quadrature_points);

            for (const auto q : fe_eval.quadrature_point_indices())
              {
                const auto point = fe_eval.quadrature_point(q);
                point_type p(point[0], point[1]);
                vec[q] = static_cast<double>(boost::geometry::within(p, poly));
              }

            cell->set_dof_values(vec, force_vector_sharp_interface);
          }
      }
    }
  } // namespace GridTools

} // namespace dealii
