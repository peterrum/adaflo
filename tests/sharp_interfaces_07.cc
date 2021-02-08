#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

template <int dim>
class LSFunction : public Function<dim>
{
public:
  LSFunction()
    : Function<dim>(1, 0)
  {}

  double
  value(const Point<dim> &p, const unsigned int) const
  {
    return p.distance(Point<dim>()) - 0.5;
  }
};

template <int dim>
class MarchingCubeAlgorithm
{
public:
  MarchingCubeAlgorithm(const Mapping<dim, dim> &      mapping,
                        const FiniteElement<dim, dim> &fe,
                        const unsigned int             n_subdivisions)
    : n_subdivisions(n_subdivisions)
    , fe_values(mapping,
                fe,
                create_qudrature_rule(n_subdivisions),
                update_values | update_quadrature_points)
    , ls_values(fe_values.n_quadrature_points)
  {
    AssertDimension(dim, 2);
  }

  template <typename CellType, typename VectorType>
  void
  process_cell(const CellType &                  cell,
               const VectorType &                ls_vector,
               std::vector<Point<dim>> &         vertices,
               std::vector<::CellData<dim - 1>> &cells)
  {
    fe_values.reinit(cell);
    fe_values.get_function_values(ls_vector, ls_values);

    for (unsigned int j = 0; j < n_subdivisions; ++j)
      for (unsigned int i = 0; i < n_subdivisions; ++i)
        {
          std::vector<unsigned int> mask{(n_subdivisions + 1) * (j + 0) + (i + 0),
                                         (n_subdivisions + 1) * (j + 0) + (i + 1),
                                         (n_subdivisions + 1) * (j + 1) + (i + 1),
                                         (n_subdivisions + 1) * (j + 1) + (i + 0),
                                         (n_subdivisions + 1) * (n_subdivisions + 1) +
                                           (n_subdivisions * j + i)};

          process_sub_cell(
            ls_values, fe_values.get_quadrature_points(), mask, vertices, cells);
        }
  }

private:
  static Quadrature<dim>
  create_qudrature_rule(const unsigned int n_subdivisions)
  {
    std::vector<Point<dim>> quadrature_points;

    for (unsigned int j = 0; j <= n_subdivisions; ++j)
      for (unsigned int i = 0; i <= n_subdivisions; ++i)
        quadrature_points.emplace_back(1.0 / n_subdivisions * i,
                                       1.0 / n_subdivisions * j);

    for (unsigned int j = 0; j < n_subdivisions; ++j)
      for (unsigned int i = 0; i < n_subdivisions; ++i)
        quadrature_points.emplace_back(1.0 / n_subdivisions * (i + 0.5),
                                       1.0 / n_subdivisions * (j + 0.5));

    return {quadrature_points};
  }

  static void
  process_sub_cell(const std::vector<double> &       ls_values,
                   const std::vector<Point<dim>> &   points,
                   const std::vector<unsigned int>   mask,
                   std::vector<Point<dim>> &         vertices,
                   std::vector<::CellData<dim - 1>> &cells)
  {
    unsigned int c = 0;

    for (unsigned int i = 0, scale = 1; i < 4; ++i, scale *= 2)
      c += (ls_values[mask[i]] > 0) * scale;

    if (c == 0 || c == 15)
      return; // nothing to do

    const auto process_points = [&](const auto &lines) {
      const double w0 = std::abs(ls_values[mask[lines[0]]]);
      const double w1 = std::abs(ls_values[mask[lines[1]]]);

      return points[mask[lines[0]]] * (w1 / (w0 + w1)) +
             points[mask[lines[1]]] * (w0 / (w0 + w1));
    };

    const auto process_lines = [&](const auto &lines) {
      std::array<std::array<unsigned int, 2>, 4> table{
        {{{0, 3}}, {{1, 2}}, {{0, 1}}, {{3, 2}}}};

      const auto p0 = process_points(table[lines[0]]);
      const auto p1 = process_points(table[lines[1]]);

      cells.resize(cells.size() + 1);
      cells.back().vertices[0] = vertices.size();
      cells.back().vertices[1] = vertices.size() + 1;

      vertices.emplace_back(p0);
      vertices.emplace_back(p1);
    };

    if (c == 5 || c == 10)
      {
        Assert(false, ExcNotImplemented());
        return;
      }

    static const unsigned int X = -1;

    std::array<std::array<unsigned int, 2>, 16> table{{
      {{X, X}},
      {{0, 2}},
      {{1, 2}},
      {{0, 1}}, //  0- 3
      {{1, 3}},
      {{X, X}},
      {{2, 3}},
      {{0, 3}}, //  4- 7
      {{0, 3}},
      {{2, 3}},
      {{X, X}},
      {{1, 3}}, //  8-11
      {{0, 1}},
      {{2, 1}},
      {{0, 2}},
      {{X, X}} // 12-15
    }};

    process_lines(table[c]);
  }

  const unsigned int  n_subdivisions;
  FEValues<dim>       fe_values;
  std::vector<double> ls_values;
};

template <int dim>
void
test()
{
  const unsigned int n_refinements  = 2;
  const unsigned int fe_degree      = 2;
  const unsigned int n_subdivisions = 3;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>{fe_degree});

  Vector<double> ls_vector(dof_handler.n_dofs());

  MappingQGeneric<dim> mapping(1);

  VectorTools::interpolate(mapping, dof_handler, LSFunction<dim>(), ls_vector);

  std::vector<Point<dim>>          vertices;
  std::vector<::CellData<dim - 1>> cells;

  MarchingCubeAlgorithm<dim> mc(mapping, dof_handler.get_fe(), n_subdivisions);

  for (const auto &cell : dof_handler.active_cell_iterators())
    mc.process_cell(cell, ls_vector, vertices, cells);

  Triangulation<dim - 1, dim> tria_interface;
  tria_interface.create_triangulation(vertices, cells, {});

  {
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;

    DataOut<dim> data_out;
    data_out.set_flags(flags);

    data_out.add_data_vector(dof_handler, ls_vector, "ls");
    data_out.build_patches(mapping, 4);

    std::ofstream out("sharp_interfaces_07_a.vtk");
    data_out.write_vtk(out);
  }

  {
    std::ofstream out("sharp_interfaces_07_b.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria_interface, out);
  }
}

int
main()
{
  test<2>();
}